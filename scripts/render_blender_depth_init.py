#!/usr/bin/env python3
"""Render depth-only training views and build a RadiosityGS point initializer.

Run with Blender against the *original*, unnormalised blend file.  Camera
poses are loaded from an existing RadiosityGS/GS^3 Blender dataset, so this
does not replace its RGB images or transforms:

    blender -b ~/data/Cycles.blend --python scripts/render_blender_depth_init.py -- \
        --dataset ~/data/Cycles_radiositygs

The script applies exactly the same unit-cube normalization as
render_blender_radiositygs.py, renders the Blender Z pass for training views,
and back-projects a bounded number of valid pixels from each depth image into
dataset/points3d.ply.  This is a geometry-assisted initializer: it is useful
for a synthetic Blender scene, but is not comparable to image-only/SfM
initialization.
"""

import argparse
from array import array
import json
import math
import random
import shutil
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Reuse the exact bounds/normalization implementation used to create the RGB
# dataset.  The source blend is reopened for this script, so mutating it only
# affects Blender's in-memory copy.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from render_blender_radiositygs import (  # noqa: E402
    evaluated_bounds,
    normalize_scene,
    remove_existing_cameras_and_lights,
)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render Blender Z maps and make depth-backed points3d.ply."
    )
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Existing dataset containing transforms_train.json")
    parser.add_argument("--samples", type=int, default=2,
                        help="Cycles samples per depth view (default: 2)")
    parser.add_argument("--resolution", type=int, default=0,
                        help="Square depth resolution; 0 reuses dataset_info.json")
    parser.add_argument("--points-per-view", type=int, default=512,
                        help="Maximum valid depth pixels retained from each training view")
    parser.add_argument("--max-depth", type=float, default=20.0,
                        help="Discard Z values outside (0, max-depth)")
    parser.add_argument("--color-init", choices=("rgb", "gray"), default="rgb",
                        help="Use existing HDR RGB samples or neutral gray PLY colors")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-views", type=int, default=0,
                        help="Only process this many train views (0 means all; useful to test)")
    parser.add_argument("--device", choices=("AUTO", "CPU", "CUDA", "OPTIX"),
                        default="AUTO")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace dataset/depth/train and dataset/points3d.ply")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.points_per_view < 1:
        parser.error("--samples and --points-per-view must be positive")
    if args.resolution < 0 or args.max_depth <= 0 or args.max_views < 0:
        parser.error("--resolution, --max-depth, and --max-views are invalid")
    return args


def dataset_resolution(dataset, requested):
    if requested:
        return requested
    info_path = dataset / "dataset_info.json"
    if not info_path.exists():
        raise RuntimeError("--resolution is required when dataset_info.json is absent")
    with info_path.open("r", encoding="utf-8") as handle:
        resolution = json.load(handle).get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2 or resolution[0] != resolution[1]:
        raise RuntimeError("dataset_info.json has no square resolution; pass --resolution")
    return int(resolution[0])


def configure_cycles_depth(scene, args, resolution):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.use_file_extension = True

    if args.device == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if args.device != "AUTO":
            prefs.compute_device_type = args.device
        prefs.get_devices()
        enabled = []
        for device in prefs.devices:
            use = (device.type in {"OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"}
                   if args.device == "AUTO" else device.type == args.device)
            device.use = use
            if use:
                enabled.append(device.name)
        if enabled:
            scene.cycles.device = "GPU"
            return ", ".join(enabled)
        if args.device != "AUTO":
            raise RuntimeError("no {} Cycles device found".format(args.device))
    except Exception as exc:
        if args.device != "AUTO":
            raise
        print("[DepthInit] GPU auto-detection failed; using CPU: {}".format(exc))
    scene.cycles.device = "CPU"
    return "CPU"


def create_camera(scene, camera_angle_x):
    data = bpy.data.cameras.new("RadiosityGS_DepthCamera")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.angle = camera_angle_x
    data.lens_unit = "FOV"
    data.clip_start = 0.01
    data.clip_end = 100.0
    camera = bpy.data.objects.new("RadiosityGS_DepthCamera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def configure_depth_output(scene, depth_dir):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layer = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.base_path = str(depth_dir)
    output.format.file_format = "OPEN_EXR"
    # Blender's OpenEXR file-output node accepts RGB/RGBA rather than BW on
    # older releases; connecting one scalar Depth socket replicates it into
    # the RGB channels, and the back-projection below reads the first one.
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    output.format.exr_codec = "ZIP"
    tree.links.new(layer.outputs["Depth"], output.inputs[0])
    scene.view_layers[0].use_pass_z = True
    return output


def read_image_pixels(path):
    """Return (width, height, channels, float pixels) and unload the image later."""
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size[:]
    channels = image.channels
    pixels = array("f", [0.0]) * (width * height * channels)
    image.pixels.foreach_get(pixels)
    return image, width, height, channels, pixels


def frame_rgb_pixels(dataset, frame, expected_width, expected_height):
    source = dataset / (frame["file_path"].lstrip("./") + ".exr")
    if not source.exists():
        raise RuntimeError("missing RGB EXR required by --color-init rgb: {}".format(source))
    image, width, height, channels, pixels = read_image_pixels(source)
    if (width, height) != (expected_width, expected_height):
        bpy.data.images.remove(image)
        raise RuntimeError("RGB/depth resolution mismatch for {}".format(source))
    return image, channels, pixels


def collect_points_from_depth(depth_path, frame, dataset, camera, args, rng):
    depth_image, width, height, depth_channels, depths = read_image_pixels(depth_path)
    valid = []
    for pixel_idx in range(width * height):
        depth = depths[pixel_idx * depth_channels]
        if math.isfinite(depth) and 0.0 < depth < args.max_depth:
            valid.append(pixel_idx)
    if not valid:
        bpy.data.images.remove(depth_image)
        raise RuntimeError("no valid depth pixels in {}".format(depth_path))
    selected = rng.sample(valid, min(args.points_per_view, len(valid)))

    rgb_image = rgb_channels = rgb_pixels = None
    if args.color_init == "rgb":
        rgb_image, rgb_channels, rgb_pixels = frame_rgb_pixels(
            dataset, frame, width, height)

    focal_x = width / (2.0 * math.tan(camera.data.angle * 0.5))
    focal_y = focal_x
    c2w = camera.matrix_world.copy()
    origin = c2w.translation
    points = []
    for pixel_idx in selected:
        u = pixel_idx % width
        v = pixel_idx // width  # Blender image buffers start at the bottom row.
        depth = depths[pixel_idx * depth_channels]
        ray_camera = Vector(((u + 0.5 - width * 0.5) / focal_x,
                             (v + 0.5 - height * 0.5) / focal_y,
                             -1.0)).normalized()
        point = origin + (c2w.to_3x3() @ ray_camera) * depth
        if args.color_init == "rgb":
            offset = pixel_idx * rgb_channels
            rgb = [max(0.0, min(1.0, rgb_pixels[offset + channel]))
                   for channel in range(3)]
        else:
            rgb = [0.5, 0.5, 0.5]
        points.append((point.x, point.y, point.z, *rgb))

    bpy.data.images.remove(depth_image)
    if rgb_image is not None:
        bpy.data.images.remove(rgb_image)
    return points, len(valid)


def write_ply(path, points):
    if not points:
        raise RuntimeError("cannot write an empty point cloud")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Depth-backprojected training-view initialization\n"
        "element vertex {}\n".format(len(points))
        + "property float x\nproperty float y\nproperty float z\n"
        + "property float nx\nproperty float ny\nproperty float nz\n"
        + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    record = struct.Struct("<ffffffBBB")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for x, y, z, r, g, b in points:
            handle.write(record.pack(
                x, y, z, 0.0, 0.0, 0.0,
                round(255.0 * r), round(255.0 * g), round(255.0 * b)))


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    transforms_path = dataset / "transforms_train.json"
    if not transforms_path.exists():
        raise RuntimeError("dataset must contain transforms_train.json: {}".format(dataset))
    with transforms_path.open("r", encoding="utf-8") as handle:
        transforms = json.load(handle)
    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError("transforms_train.json contains no frames")
    if args.max_views:
        frames = frames[:args.max_views]
    if "camera_angle_x" not in transforms:
        raise RuntimeError("transforms_train.json has no camera_angle_x")

    depth_dir = dataset / "depth" / "train"
    point_path = dataset / "points3d.ply"
    if (depth_dir.exists() or point_path.exists()) and not args.overwrite:
        raise RuntimeError(
            "depth/train or points3d.ply already exists; pass --overwrite to replace only those outputs")
    if depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    original_lo, original_hi, _ = evaluated_bounds(scene)
    remove_existing_cameras_and_lights()
    center, scale, _ = normalize_scene(scene, original_lo, original_hi, 0.90)
    resolution = dataset_resolution(dataset, args.resolution)
    device = configure_cycles_depth(scene, args, resolution)
    camera = create_camera(scene, float(transforms["camera_angle_x"]))
    output_node = configure_depth_output(scene, depth_dir)
    rng = random.Random(args.seed)
    all_points = []

    print("[DepthInit] normalized center={}, scale={:.9g}".format(tuple(center), scale))
    print("[DepthInit] Cycles device={}, samples/view={}, resolution={}x{}".format(
        device, args.samples, resolution, resolution))
    for index, frame in enumerate(frames):
        camera.matrix_world = Matrix(frame["transform_matrix"])
        bpy.context.view_layer.update()
        stem = Path(frame["file_path"]).name
        output_node.file_slots[0].path = stem
        print("[DepthInit] rendering depth {}/{} ({})".format(index + 1, len(frames), stem))
        bpy.ops.render.render(write_still=False)
        # Blender's File Output node appends the current frame number.  Keep
        # that stable Blender-native name rather than relying on version-
        # dependent filename conventions.
        written = sorted(depth_dir.glob(stem + "*.exr"),
                         key=lambda path: path.stat().st_mtime_ns)
        if not written:
            raise RuntimeError("Blender did not write a depth EXR for {}".format(stem))
        depth_path = written[-1]
        points, valid_count = collect_points_from_depth(
            depth_path, frame, dataset, camera, args, rng)
        all_points.extend(points)
        print("[DepthInit] {} valid pixels; retained {} points".format(
            valid_count, len(points)))

    write_ply(point_path, all_points)
    print("[DepthInit] wrote {} points to {}".format(len(all_points), point_path))


if __name__ == "__main__":
    main()
