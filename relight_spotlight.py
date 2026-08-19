#
# relight_spotlight.py
#
# Relight a *trained* RadiosityGS scene with a SPOTLIGHT instead of the point
# light it was captured with. Nothing is optimized here -- geometry, materials and
# the spot are all fixed inputs; this just renders.
#
# The spotlight is a normal point light whose angular emission profile has been
# baked into its SH emission bands (see scene/spot_light.py). No CUDA change, no
# recompilation, no new derivatives.
#
#   factor(theta) = 1                                     if theta <= --cutoff_deg
#                 = exp(-(theta - cutoff)^2 / 2 sigma^2)  otherwise
#
# Three ways to drive it:
#   --sweep fixed   one render, the spot exactly where you put it
#   --sweep orbit   move the light around the object on a circle, always aimed at
#                   the target  ->  "relight from different positions"
#   --sweep aim     keep the light still and swing the cone across the scene
#                   ->  "relight from different angles"
#
# Runs on the GPU node (needs CUDA / the compiled radiosity solver).
#
# Examples
# --------
#   # single spotlight, placed like the dataset's mean light, aimed at the object
#   python relight_spotlight.py -m output/50_Hotdog --cutoff_deg 20 --sigma_deg 12
#
#   # 24-frame orbit of the spot around the object, 1.5 object-radii up
#   python relight_spotlight.py -m output/50_Hotdog --sweep orbit --frames 24 \
#       --orbit_radius 2.5 --orbit_elevation 40
#
#   # fixed light, cone swinging 60 deg across the scene
#   python relight_spotlight.py -m output/50_Hotdog --sweep aim --frames 16 --aim_range 60
#

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import math
from time import strftime, localtime
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.light_source import LightModel
from scene.spot_light import SpotLight, ascii_profile
from gaussian_renderer import renderGI
from arguments import PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.render_utils import save_img_u8
from utils.sh_utils import SH2RGB
from render import read_cfg


def orthonormal_basis(axis: torch.Tensor):
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    axis = torch.nn.functional.normalize(axis.reshape(3), dim=0)
    helper = torch.tensor([0., 0., 1.], device=axis.device) if abs(axis[2]) < 0.9 \
        else torch.tensor([1., 0., 0.], device=axis.device)
    u = torch.nn.functional.normalize(torch.linalg.cross(axis, helper), dim=0)
    return u, torch.linalg.cross(axis, u)


def main():
    parser = ArgumentParser(description="Relight a trained RadiosityGS scene with a spotlight")
    pipeline = PipelineParams(parser)
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to the trained model (the output/... folder)")
    parser.add_argument("--iteration", type=int, default=-1, help="Which saved iteration to load (-1 = latest)")
    parser.add_argument("--num_walks", type=int, default=128, help="Monte-Carlo walks for the (hybrid/MC) solver")

    # --- the spot ---
    parser.add_argument("--cutoff_deg", type=float, default=20.,
                        help="Half-angle of the fully-lit inner cone (factor = 1 inside)")
    parser.add_argument("--sigma_deg", type=float, default=12.,
                        help="Gaussian roll-off width outside the cone. Keep cutoff+sigma >= ~20 deg: "
                             "the SH emission is band-limited to degree 9 (~20 deg of angular detail), "
                             "so tighter cones come out softer than requested (the printed fit tells you how much)")
    parser.add_argument("--fit_sh_degree", type=int, default=-1,
                        help="SH degree used for the profile (-1 = the model's active degree). "
                             "Lower = smoother cone, less ringing")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Smoothness weight of the SH fit")

    # --- placement ---
    parser.add_argument("--light_pos", type=float, nargs=3, default=None,
                        help="World-space light position (default: the dataset's mean pl_pos)")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        help="World-space point the cone is aimed at (default: the object's center)")
    parser.add_argument("--intensity", type=float, nargs=3, default=None,
                        help="On-axis linear RGB radiance (default: the dataset's mean pl_intensity)")
    parser.add_argument("--intensity_scale", type=float, default=1.,
                        help="Multiplies --intensity. A spot concentrates the same nominal intensity "
                             "into a cone, so the lit patch is as bright as the point light was there")

    # --- sweeps ---
    parser.add_argument("--sweep", type=str, default="fixed", choices=["fixed", "orbit", "aim"])
    parser.add_argument("--frames", type=int, default=1, help="Number of renders in the sweep")
    parser.add_argument("--orbit_radius", type=float, default=-1.,
                        help="[orbit] Light distance from the target, in object radii (-1 = keep --light_pos's distance)")
    parser.add_argument("--orbit_elevation", type=float, default=30.,
                        help="[orbit] Elevation above the orbit plane, in degrees")
    parser.add_argument("--orbit_axis", type=float, nargs=3, default=[0., 0., 1.],
                        help="[orbit] World axis the light orbits around")
    parser.add_argument("--orbit_degrees", type=float, default=360., help="[orbit] Total arc swept")
    parser.add_argument("--aim_range", type=float, default=60.,
                        help="[aim] Total angle the cone swings across, in degrees")
    parser.add_argument("--aim_axis", type=float, nargs=3, default=[0., 0., 1.],
                        help="[aim] World axis the cone direction rotates around")

    # --- output ---
    parser.add_argument("--view_split", type=str, default="auto", choices=["auto", "train", "test"])
    parser.add_argument("--view_index", type=int, default=0, help="Which camera to render from")
    parser.add_argument("--all_views", action="store_true",
                        help="Render every camera of the split instead of a sweep of one camera")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Where to write the renders (default: <model_path>/spotlight_relight/<timestamp>)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    # ---- Load the trained scene ----
    dataset = read_cfg(args.model_path)
    pipe = pipeline.extract(args)
    if pipe.compute_cov3D_python:
        # That path asks the light source for get_covariance(), which no light
        # source in this repo implements (LightModel does not either).
        raise SystemExit("[ABORT] --compute_cov3D_python is not supported for light sources.")

    gaussians = GaussianModel(dataset)
    placeholder = LightModel(dataset)  # only so Scene can load; discarded
    placeholder.create_from_env_map(init_intensity=1.)
    scene = Scene(dataset, gaussians, placeholder, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams, test_cams = scene.getTrainCameras(), scene.getTestCameras()
    if args.view_split == "test" or (args.view_split == "auto" and len(test_cams) > 0):
        cams, split_name = test_cams, "test"
    else:
        cams, split_name = train_cams, "train"

    center = gaussians.get_xyz.detach().mean(dim=0)
    obj_radius = (gaussians.get_xyz.detach() - center).norm(dim=1).max()

    # ---- Defaults taken from the dataset's own point lights ----
    lit_cams = [c for c in train_cams if c.pl_pos is not None]
    if args.light_pos is None or args.intensity is None:
        if len(lit_cams) == 0:
            raise SystemExit("[ABORT] This scene has no per-camera point lights to take defaults from; "
                             "pass --light_pos and --intensity explicitly.")
        pl_positions = torch.stack([c.pl_pos.detach() for c in lit_cams])
        pl_intensities = torch.stack([c.pl_intensity.detach().reshape(3) for c in lit_cams])

    light_pos = torch.tensor(args.light_pos, dtype=torch.float32, device="cuda") \
        if args.light_pos is not None else pl_positions.mean(dim=0)
    target = torch.tensor(args.target, dtype=torch.float32, device="cuda") \
        if args.target is not None else center
    intensity = torch.tensor(args.intensity, dtype=torch.float32, device="cuda") \
        if args.intensity is not None else SH2RGB(pl_intensities.mean(dim=0))
    intensity = intensity * args.intensity_scale

    fit_degree = gaussians.active_sh_degree if args.fit_sh_degree < 0 else args.fit_sh_degree
    if fit_degree > gaussians.active_sh_degree:
        # renderGI hands pc.active_sh_degree to the solver, which silently ignores
        # every band above it -- a higher-degree fit would be truncated to garbage.
        raise SystemExit(f"[ABORT] --fit_sh_degree {fit_degree} exceeds the model's active SH degree "
                         f"{gaussians.active_sh_degree}; those bands would be ignored by the solver.")

    spot = SpotLight(
        position=light_pos,
        direction=target - light_pos,
        intensity=intensity,
        cutoff_deg=args.cutoff_deg,
        sigma_deg=args.sigma_deg,
        max_sh_degree=gaussians.max_sh_degree,
        fit_sh_degree=fit_degree,
        ridge=args.ridge,
    )

    print(f"\nObject center / radius: {center.tolist()} / {obj_radius.item():.4f}")
    print(f"Aiming at:              {target.tolist()}")
    print(f"{spot.describe()}\n")
    print(ascii_profile(spot.fit))
    print("\n  '.' = requested profile, '#' = what degree-"
          f"{fit_degree} SH can actually emit."
          "\n  The lobe past 90 deg is the mirror cone: the solver folds emission directions into one"
          "\n  hemisphere, so every spotlight is double-ended. Aim it so the back lobe hits nothing.\n")

    solver_settings = {
        "inverse_falloff_max": dataset.max_inverse_falloff,
        "num_walks": args.num_walks,
        "gradient_num_walks": args.num_walks,
        "min_decay": 1e-4,
        "use_cluster": not pipe.not_use_cluster,
    }

    base_dir = args.output_dir or os.path.join(args.model_path, "spotlight_relight")
    out_dir = os.path.join(base_dir, strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing renders to: {out_dir}\n")

    # ---- Build the (camera, light placement) list ----
    if args.all_views:
        jobs = [(cam, light_pos, target, f"{split_name}_{i:04d}_{cam.image_name}")
                for i, cam in enumerate(cams)]
    elif args.sweep == "fixed":
        cam = cams[args.view_index % len(cams)]
        jobs = [(cam, light_pos, target, f"{split_name}{args.view_index % len(cams)}_fixed")]
    else:
        cam = cams[args.view_index % len(cams)]
        frames = max(args.frames, 1)
        jobs = []
        if args.sweep == "orbit":
            axis = torch.tensor(args.orbit_axis, dtype=torch.float32, device="cuda")
            axis = torch.nn.functional.normalize(axis, dim=0)
            u, v = orthonormal_basis(axis)
            radius = (light_pos - target).norm() if args.orbit_radius < 0 else obj_radius * args.orbit_radius
            elev = math.radians(args.orbit_elevation)
            for f in range(frames):
                phi = math.radians(args.orbit_degrees) * f / frames
                offset = (math.cos(phi) * math.cos(elev) * u
                          + math.sin(phi) * math.cos(elev) * v
                          + math.sin(elev) * axis)
                jobs.append((cam, target + radius * offset, target, f"orbit_{f:04d}"))
        else:  # aim
            axis = torch.nn.functional.normalize(
                torch.tensor(args.aim_axis, dtype=torch.float32, device="cuda"), dim=0)
            base_dir_vec = torch.nn.functional.normalize(target - light_pos, dim=0)
            reach = (target - light_pos).norm()
            for f in range(frames):
                # Rotate the aim direction about `axis` (Rodrigues), sweeping
                # symmetrically about the base direction.
                a = math.radians(args.aim_range) * (f / max(frames - 1, 1) - 0.5)
                d = (base_dir_vec * math.cos(a)
                     + torch.linalg.cross(axis, base_dir_vec) * math.sin(a)
                     + axis * torch.dot(axis, base_dir_vec) * (1. - math.cos(a)))
                jobs.append((cam, light_pos, light_pos + reach * torch.nn.functional.normalize(d, dim=0),
                             f"aim_{f:04d}"))

    # ---- Render ----
    with torch.no_grad():
        for cam, pos, aim, name in tqdm(jobs, desc="Relighting"):
            spot.place(pos, aim)
            pkg = renderGI(cam, gaussians, spot, pipe, background,
                           override_solver_settings=solver_settings)
            save_img_u8(pkg["render"].clamp(0., 1.).permute(1, 2, 0).cpu().numpy(),
                        os.path.join(out_dir, f"{name}.png"))

    with open(os.path.join(out_dir, "spotlight.txt"), "w") as f:
        f.write(spot.describe() + "\n\n" + ascii_profile(spot.fit) + "\n")
    print(f"\nDone. {len(jobs)} render(s) in {out_dir}")


if __name__ == "__main__":
    main()
