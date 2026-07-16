#
# optimize_light.py
#
# Take a *trained* RadiosityGS scene, FREEZE all geometry/material (the surfels),
# throw away the light, and re-solve ONLY a single point light from a random
# initialization. Renders a fixed camera view at the start (random light) and
# every few epochs so you can watch the light converge / "move" in the scene.
#
# This scene uses a single point light (GS^3-style pl_pos / pl_intensity stored
# per camera). Those are fixed input data, not trainable parameters, so here we
# introduce ONE global learnable point light (position + RGB intensity) and
# optimize it against the frozen scene.
#
# Runs on the GPU node (needs CUDA / the compiled radiosity solver). Nothing here
# is meant to run on a CPU-only laptop.
#

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import random
from argparse import ArgumentParser

import torch
from torch import nn
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.light_source import LightModel
from gaussian_renderer import renderGI
from arguments import PipelineParams, get_combined_args
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from utils.render_utils import save_img_u8
from render import read_cfg


class LearnablePointLight:
    """A single, global, trainable point light.

    Mirrors the interface that ``renderGI`` expects from a light-source object
    (see ``LightModel.from_camera_if_possible``). Only the position and the RGB
    (SH DC) intensity are optimizable; everything else matches the constants a
    per-camera point light uses.
    """

    def __init__(self, max_sh_degree, init_xyz, init_intensity_sh, device="cuda"):
        self.max_sh_degree = max_sh_degree
        self.is_directional_light = False
        self._device = device
        # (1, 3) world-space position, and (1, 3) SH DC intensity (RGB2SH space,
        # same space as camera.pl_intensity / LightModel._intensity).
        self._xyz = nn.Parameter(init_xyz.reshape(1, 3).float().to(device).contiguous())
        self._intensity = nn.Parameter(init_intensity_sh.reshape(1, 3).float().to(device).contiguous())

    # --- trainable ---
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_emissions(self):
        # (1, (deg+1)^2, 3): only the DC band emits; higher bands are zero.
        dc = torch.relu(self._intensity)[:, None, :]  # (1, 1, 3)
        n_rest = (self.max_sh_degree + 1) ** 2 - 1
        rest = torch.zeros((1, n_rest, 3), dtype=dc.dtype, device=self._device)
        return torch.cat((dc, rest), dim=1)

    # --- fixed constants (match a per-camera point light) ---
    @property
    def get_geovalue(self):
        return torch.tensor([6.], device=self._device)[None]

    @property
    def get_norm_factor(self):
        return torch.tensor([1.], device=self._device)[None]

    @property
    def get_scaling(self):
        return torch.tensor([1e-3, 1e-3], device=self._device)[None]

    @property
    def get_rotation(self):
        return torch.tensor([1., 0., 0., 0.], device=self._device)[None]

    @property
    def get_is_light_source(self):
        return torch.tensor([True], device=self._device)[None]

    def parameters(self):
        return [self._xyz, self._intensity]


def freeze_gaussians(gaussians: GaussianModel):
    for attr in ["_xyz", "_scaling", "_rotation", "_geovalue", "_norm_factor",
                 "_blending", "_shininess", "_diffuse_albedos", "_specular_albedos"]:
        p = getattr(gaussians, attr, None)
        if p is not None:
            p.requires_grad_(False)


def gt_of(cam):
    """Ground-truth image (3, H, W) on cuda, matching how train.py picks it."""
    if cam.exr_image is not None:
        img = cam.exr_image[:3]
    else:
        img = cam.original_image
    return img.cuda()


def alpha_of(cam, ref):
    if cam.gt_alpha_mask is not None:
        return cam.gt_alpha_mask.cuda()
    return torch.ones_like(ref[:1])


def main():
    parser = ArgumentParser(description="Re-solve a single point light for a trained RadiosityGS scene")
    pipeline = PipelineParams(parser)
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to the trained model (the output/... folder)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Which saved iteration to load (-1 = latest)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of passes over the training views")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save a fixed-view render every N epochs (epoch 0 always saved)")
    parser.add_argument("--num_walks", type=int, default=64,
                        help="Monte-Carlo walks for the (hybrid/MC) solver")
    parser.add_argument("--position_lr", type=float, default=0.01)
    parser.add_argument("--intensity_lr", type=float, default=0.05)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    # Fixed camera used for the epoch snapshots.
    parser.add_argument("--view_split", type=str, default="auto", choices=["auto", "train", "test"])
    parser.add_argument("--view_index", type=int, default=0)
    # Random light initialization.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_dist_scale", type=float, default=2.0,
                        help="Init light distance from object center, in units of the object's radius")
    parser.add_argument("--random_intensity", action="store_true",
                        help="Also randomize the initial intensity (default: seed it from the dataset mean)")
    parser.add_argument("--init_intensity", type=float, default=1.0,
                        help="RGB scale used when --random_intensity is set")
    parser.add_argument("--max_views", type=int, default=-1,
                        help="Optimize against at most this many training views per epoch (-1 = all)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Where to write light_optimization/ (default: inside the model path)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Load the trained scene (frozen) ----
    dataset = read_cfg(args.model_path)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset)
    light_sources = LightModel(dataset)  # only needed so Scene can load; we discard it
    light_sources.create_from_env_map(init_intensity=1.)
    scene = Scene(dataset, gaussians, light_sources, load_iteration=args.iteration, shuffle=False)
    freeze_gaussians(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    # ---- Sanity check: this must be a point-light scene ----
    lit_cams = [c for c in train_cams if c.pl_pos is not None]
    if len(lit_cams) == 0:
        print("\n[ABORT] The training cameras have no point lights (pl_pos is None).")
        print("        This looks like an environment-map scene, so there is no point")
        print("        light to solve. Optimize LightModel._intensity instead.")
        sys.exit(1)

    pl_positions = torch.stack([c.pl_pos.detach() for c in lit_cams])  # (V, 3)
    pos_std = pl_positions.std(dim=0)
    print(f"\nPoint light present in {len(lit_cams)}/{len(train_cams)} train views.")
    print(f"Dataset pl_pos  mean = {pl_positions.mean(0).tolist()}")
    print(f"Dataset pl_pos  std  = {pos_std.tolist()}  "
          f"(near-zero std => a single global light; large std => per-view/OLAT lights)")

    # ---- Randomly initialize the single learnable point light ----
    center = gaussians.get_xyz.detach().mean(dim=0)
    obj_radius = (gaussians.get_xyz.detach() - center).norm(dim=1).max()
    direction = torch.randn(3, device="cuda")
    direction = direction / direction.norm()
    init_xyz = center + direction * obj_radius * args.init_dist_scale

    if args.random_intensity:
        init_intensity_sh = torch.rand(3, device="cuda") * args.init_intensity * 0.28209479  # RGB2SH C0
    else:
        # Seed intensity from the dataset's average so brightness is well-scaled;
        # only the *position* is "forgotten". Add --random_intensity to forget both.
        pl_ints = torch.stack([c.pl_intensity.detach().reshape(3) for c in lit_cams])  # (V, 3) SH DC
        init_intensity_sh = pl_ints.mean(dim=0)

    point_light = LearnablePointLight(gaussians.max_sh_degree, init_xyz, init_intensity_sh)
    print(f"\nInitial light position:  {point_light._xyz.detach().reshape(3).tolist()}")
    print(f"Object center / radius:  {center.tolist()} / {obj_radius.item():.4f}")

    optimizer = torch.optim.Adam([
        {"params": [point_light._xyz], "lr": args.position_lr, "name": "position"},
        {"params": [point_light._intensity], "lr": args.intensity_lr, "name": "intensity"},
    ], eps=1e-15)

    solver_settings = {
        "inverse_falloff_max": dataset.max_inverse_falloff,
        "num_walks": args.num_walks,
        "gradient_num_walks": args.num_walks,
        "min_decay": 1e-4,
        "use_cluster": not pipe.not_use_cluster,
    }

    # ---- Fixed camera for the snapshots ----
    if args.view_split == "test" or (args.view_split == "auto" and len(test_cams) > 0):
        fixed_pool = test_cams
        split_name = "test"
    else:
        fixed_pool = train_cams
        split_name = "train"
    fixed_cam = fixed_pool[args.view_index % len(fixed_pool)]
    print(f"\nFixed snapshot view: {split_name}[{args.view_index % len(fixed_pool)}] "
          f"(image_name={fixed_cam.image_name})")

    out_dir = args.output_dir or os.path.join(args.model_path, "light_optimization")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing snapshots to: {out_dir}\n")

    # Save the ground-truth of the fixed view once, as the convergence target.
    fixed_gt = gt_of(fixed_cam).clamp(0., 1.)
    save_img_u8(fixed_gt.permute(1, 2, 0).cpu().numpy(),
                os.path.join(out_dir, "gt_reference.png"))

    @torch.no_grad()
    def save_snapshot(epoch):
        pkg = renderGI(fixed_cam, gaussians, point_light, pipe, background,
                       override_solver_settings=solver_settings)
        img = pkg["render"].clamp(0., 1.)
        save_img_u8(img.permute(1, 2, 0).cpu().numpy(),
                    os.path.join(out_dir, f"epoch_{epoch}.png"))
        p = psnr(img, fixed_gt).mean().item()
        pos = point_light._xyz.detach().reshape(3).tolist()
        print(f"[epoch {epoch:>3}] saved epoch_{epoch}.png | fixed-view PSNR {p:5.2f} | "
              f"light pos [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

    # Epoch 0 == random light, before any optimization.
    save_snapshot(0)

    n_views = len(train_cams) if args.max_views < 0 else min(args.max_views, len(train_cams))
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_cams)))
        random.shuffle(order)
        order = order[:n_views]

        pbar = tqdm(order, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for idx in pbar:
            cam = train_cams[idx]
            pkg = renderGI(cam, gaussians, point_light, pipe, background,
                           override_solver_settings=solver_settings)
            image = pkg["render"]

            gt = gt_of(cam)
            mask = alpha_of(cam, gt)
            image = image * mask  # compare on the masked object region
            gt = gt * mask

            l1 = l1_loss(image, gt)
            loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ssim(image, gt))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_snapshot(epoch)

    print(f"\nDone. See {out_dir}/epoch_*.png (and gt_reference.png).")


if __name__ == "__main__":
    main()
