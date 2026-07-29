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
from time import strftime, localtime
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
from utils.sh_utils import RGB2SH, SH2RGB
from render import read_cfg


def _inv_softplus(y):
    """Inverse of softplus, so softplus(_inv_softplus(y)) == y for y > 0."""
    y = y.clamp_min(1e-6)
    return torch.log(torch.expm1(y).clamp_min(1e-6))


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
        # Intensity is stored pre-softplus so emission stays strictly positive
        # with a live gradient everywhere (a plain relu can get stuck at 0 and
        # never recover, permanently darkening the light).
        raw_intensity = _inv_softplus(init_intensity_sh.reshape(1, 3).float().to(device))
        self._xyz = nn.Parameter(init_xyz.reshape(1, 3).float().to(device).contiguous())
        self._intensity = nn.Parameter(raw_intensity.contiguous())

    # --- trainable ---
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_dc(self):
        """Current emission DC coefficient (1, 3), strictly positive."""
        return torch.nn.functional.softplus(self._intensity)

    @property
    def get_emissions(self):
        # (1, (deg+1)^2, 3): only the DC band emits; higher bands are zero.
        dc = self.get_dc[:, None, :]  # (1, 1, 3)
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
    parser.add_argument("--position_lr", type=float, default=0.005)
    parser.add_argument("--intensity_lr", type=float, default=0.05)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    # Fitting a single point light against a multi-light (OLAT) dataset is
    # ill-posed. Use --target_view to instead fit ONE image (correct + fast).
    parser.add_argument("--target_view", type=int, default=-1,
                        help="If >=0, optimize the light to reproduce a SINGLE train view "
                             "(index into the train split). This view is also the snapshot "
                             "camera and gt_reference. Recommended for OLAT/multi-light data.")
    parser.add_argument("--steps_per_epoch", type=int, default=-1,
                        help="Gradient steps per epoch. Default: all train views (multi-view), "
                             "or 50 (single-target mode).")
    parser.add_argument("--max_light_dist", type=float, default=4.0,
                        help="Clamp the light within this many object-radii of the center, "
                             "so it can't run off to infinity.")
    # Fixed camera used for the epoch snapshots.
    parser.add_argument("--view_split", type=str, default="auto", choices=["auto", "train", "test"])
    parser.add_argument("--view_index", type=int, default=0)
    # Random light initialization.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_mode", type=str, default="dataset", choices=["dataset", "sphere"],
                        help="'dataset' (recommended): random draw from the dataset's light "
                             "cloud (mean +/- std of pl_pos), so the light starts in a "
                             "plausible, ILLUMINATING position with live gradients. "
                             "'sphere': random direction at --init_dist_scale object-radii "
                             "(can land behind the object => dead gradient => stays dark).")
    parser.add_argument("--init_dist_scale", type=float, default=2.0,
                        help="[sphere mode] Init light distance from object center, in object-radii")
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
    obj_radius_hint = (gaussians.get_xyz.detach() - gaussians.get_xyz.detach().mean(0)).norm(dim=1).max()
    if pos_std.max() > 0.1 * obj_radius_hint and args.target_view < 0:
        print("\n[WARNING] Light position varies a lot across views => this looks like a")
        print("          MULTI-LIGHT (OLAT) dataset. A single global light cannot fit all")
        print("          views and will drift. Re-run with --target_view <i> to fit ONE image.\n")

    # ---- Randomly initialize the single learnable point light ----
    center = gaussians.get_xyz.detach().mean(dim=0)
    obj_radius = (gaussians.get_xyz.detach() - center).norm(dim=1).max()

    if args.init_mode == "dataset":
        # Draw a random-but-plausible light from where the dataset's lights live.
        # Guarantees the light starts in front of / illuminating the object, so the
        # loss has a live gradient. The specific target light is still "forgotten".
        pos_mean = pl_positions.mean(dim=0)
        init_xyz = pos_mean + torch.randn(3, device="cuda") * pos_std
    else:  # sphere
        direction = torch.randn(3, device="cuda")
        direction = direction / direction.norm()
        init_xyz = center + direction * obj_radius * args.init_dist_scale

    if args.random_intensity:
        init_intensity_sh = RGB2SH(torch.rand(3, device="cuda") * args.init_intensity)
    else:
        # Seed intensity from the dataset's average so brightness is well-scaled;
        # only the *position* is "forgotten". Add --random_intensity to forget both.
        pl_ints = torch.stack([c.pl_intensity.detach().reshape(3) for c in lit_cams])  # (V, 3) SH DC
        init_intensity_sh = pl_ints.mean(dim=0)

    point_light = LearnablePointLight(gaussians.max_sh_degree, init_xyz, init_intensity_sh)
    print(f"\nInit mode: {args.init_mode}")
    print(f"Initial light position:  {point_light._xyz.detach().reshape(3).tolist()}")
    print(f"Initial light RGB:       {SH2RGB(point_light.get_dc.detach().reshape(3)).tolist()}")
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
    # In single-target mode the snapshot view IS the target, so the render should
    # converge to exactly reproduce gt_reference.png.
    target_mode = args.target_view >= 0
    if target_mode:
        target_idx = args.target_view % len(train_cams)
        fixed_cam = train_cams[target_idx]
        split_name = f"train (single-target #{target_idx})"
        print(f"\nSingle-target mode: fitting the light to train[{target_idx}] "
              f"(image_name={fixed_cam.image_name})")
    else:
        if args.view_split == "test" or (args.view_split == "auto" and len(test_cams) > 0):
            fixed_pool = test_cams
            split_name = "test"
        else:
            fixed_pool = train_cams
            split_name = "train"
        fixed_cam = fixed_pool[args.view_index % len(fixed_pool)]
        print(f"\nFixed snapshot view: {split_name}[{args.view_index % len(fixed_pool)}] "
              f"(image_name={fixed_cam.image_name})")

    # One timestamped subfolder per run so runs don't overwrite each other.
    base_dir = args.output_dir or os.path.join(args.model_path, "light_optimization")
    run_stamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    out_dir = os.path.join(base_dir, run_stamp)
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
        rgb = SH2RGB(point_light.get_dc.detach().reshape(3)).tolist()
        print(f"[epoch {epoch:>3}] saved epoch_{epoch}.png | fixed-view PSNR {p:5.2f} | "
              f"light pos [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] | "
              f"RGB [{rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}]")

    # Clamp based on the dataset's light cloud so a valid solution is never
    # clipped; fall back to object-radii if that is somehow smaller.
    pl_dist_max = (pl_positions - center).norm(dim=1).max().item()
    max_dist = max(pl_dist_max * 1.5, (obj_radius * args.max_light_dist).item())

    def clamp_light():
        with torch.no_grad():
            d = point_light._xyz - center  # (1, 3)
            dist = d.norm()
            if dist > max_dist:
                point_light._xyz.copy_(center + d / dist * max_dist)

    # How many gradient steps per epoch, and which views they use.
    if target_mode:
        steps = args.steps_per_epoch if args.steps_per_epoch > 0 else 50
    else:
        cap = len(train_cams) if args.max_views < 0 else min(args.max_views, len(train_cams))
        steps = args.steps_per_epoch if args.steps_per_epoch > 0 else cap
    print(f"{steps} gradient steps/epoch x {args.epochs} epochs "
          f"= {steps * args.epochs} total steps. Light clamped within {max_dist:.3f} of center.\n")

    # Epoch 0 == random light, before any optimization.
    save_snapshot(0)

    for epoch in range(1, args.epochs + 1):
        if target_mode:
            batch = [target_idx] * steps
        else:
            order = list(range(len(train_cams)))
            random.shuffle(order)
            batch = order[:steps]

        pbar = tqdm(batch, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
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
            clamp_light()  # keep the light from running off to infinity
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_snapshot(epoch)

    print(f"\nDone. See {out_dir}/epoch_*.png (and gt_reference.png).")


if __name__ == "__main__":
    main()
