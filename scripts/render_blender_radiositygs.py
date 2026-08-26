#!/usr/bin/env python3
"""Render a Blender scene into RadiosityGS's GS^3-style dataset format.

Run this script with Blender, not CPython. For example:

    blender -b ~/data/Cycles.blend \
      --python scripts/render_blender_radiositygs.py -- \
      --output ~/data/Cycles_radiositygs

The source .blend is never saved. The scene is normalized in memory, lit by a
single known point light per frame, and rendered as linear RGBA OpenEXR files.
"""

import argparse
import json
import math
import random
import shutil
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Create a GS^3-style HDR dataset for RadiosityGS.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-views", type=int, default=100)
    parser.add_argument("--test-views", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--fov", type=float, default=50.0,
                        help="Horizontal field of view in degrees.")
    parser.add_argument("--layout", choices=("auto", "exterior", "cornell"),
                        default="auto",
                        help="Camera/light placement; auto detects Cornell-box scenes.")
    parser.add_argument("--camera-elevations", default="12,28,44,60,76",
                        help="Comma-separated elevation rings in degrees.")
    parser.add_argument("--light-elevations", default="15,35,55,75",
                        help="Comma-separated point-light elevation rings.")
    parser.add_argument("--light-power", type=float, default=None,
                        help="Blender/GS^3 point-light power (default: 10 Cornell, 100 exterior).")
    parser.add_argument("--light-color", default="1,1,1",
                        help="Linear RGB light color, comma separated.")
    parser.add_argument("--margin", type=float, default=0.90,
                        help="Fraction of the unit cube occupied by scene geometry.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--initial-points", type=int, default=50_000)
    parser.add_argument("--device", choices=("AUTO", "CPU", "CUDA", "OPTIX"),
                        default="AUTO")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this script's known outputs if present.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect and normalize the scene without rendering.")
    args = parser.parse_args(argv)

    try:
        args.camera_elevations = parse_csv_floats(args.camera_elevations, "camera elevations")
        args.light_elevations = parse_csv_floats(args.light_elevations, "light elevations")
        args.light_color = parse_csv_floats(args.light_color, "light color")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if len(args.light_color) != 3 or any(v < 0.0 for v in args.light_color):
        parser.error("--light-color must contain three non-negative values")
    if args.train_views < 1 or args.test_views < 1:
        parser.error("--train-views and --test-views must both be positive")
    if args.resolution < 1 or args.samples < 1 or args.initial_points < 0:
        parser.error("resolution/samples must be positive and initial-points non-negative")
    if args.light_power is not None and args.light_power <= 0.0:
        parser.error("--light-power must be positive")
    if not 0.0 < args.fov < 179.0:
        parser.error("--fov must be between 0 and 179 degrees")
    if not 0.0 < args.margin <= 1.0:
        parser.error("--margin must be in (0, 1]")
    if any(not -89.0 <= value <= 89.0
           for value in args.camera_elevations + args.light_elevations):
        parser.error("camera and light elevations must be between -89 and 89 degrees")
    return args


def parse_csv_floats(value, label):
    try:
        values = [float(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid {}: {}".format(label, exc))
    if not values:
        raise argparse.ArgumentTypeError("{} cannot be empty".format(label))
    return values


def evaluated_bounds(scene):
    """Return world-space bounds, including evaluated geometry/instances."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo = Vector((math.inf, math.inf, math.inf))
    hi = Vector((-math.inf, -math.inf, -math.inf))
    found = 0
    for instance in depsgraph.object_instances:
        obj = instance.object
        original = obj.original if hasattr(obj, "original") else obj
        if original.hide_render or obj.type not in GEOMETRY_TYPES or not hasattr(obj, "bound_box"):
            continue
        matrix = instance.matrix_world
        for corner in obj.bound_box:
            point = matrix @ Vector(corner)
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
        found += 1
    if not found or any(not math.isfinite(v) for v in (*lo, *hi)):
        raise RuntimeError("the active scene has no renderable geometry with finite bounds")
    return lo, hi, found


def remove_existing_cameras_and_lights():
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def normalize_scene(scene, lo, hi, margin):
    extent = hi - lo
    largest = max(extent)
    if largest <= 1e-12:
        raise RuntimeError("scene bounds have zero extent")
    center = (lo + hi) * 0.5
    scale = margin / largest
    normalization = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)

    # Transform only hierarchy roots; children inherit exactly once.
    roots = [obj for obj in scene.objects if obj.parent is None]
    for obj in roots:
        obj.matrix_world = normalization @ obj.matrix_world
    bpy.context.view_layer.update()
    return center, scale, normalization


def disable_unmodelled_emission(scene):
    """Remove illumination which is absent from the per-frame GS^3 metadata."""
    changed = []
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("RadiosityGS_BlackWorld")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        background.inputs["Strength"].default_value = 0.0
    world.color = (0.0, 0.0, 0.0)

    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        touched = False
        for node in material.node_tree.nodes:
            if node.type == "EMISSION" and "Strength" in node.inputs:
                if node.inputs["Strength"].default_value != 0.0:
                    node.inputs["Strength"].default_value = 0.0
                    touched = True
            elif node.type == "BSDF_PRINCIPLED":
                strength = node.inputs.get("Emission Strength")
                if strength is not None and strength.default_value != 0.0:
                    strength.default_value = 0.0
                    touched = True
        if touched:
            changed.append(material.name)
    return changed


def configure_cycles(scene, args):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.use_placeholder = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = ""
    scene.render.use_motion_blur = False
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = max(scene.cycles.max_bounces, 8)
    scene.cycles.diffuse_bounces = max(scene.cycles.diffuse_bounces, 4)
    scene.cycles.glossy_bounces = max(scene.cycles.glossy_bounces, 4)
    scene.camera = None

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
            use = device.type != "CPU"
            if args.device == "AUTO":
                use = device.type in {"OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"}
            elif args.device != "CPU":
                use = device.type == args.device
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
        print("[RadiosityGS] GPU auto-detection failed; using CPU: {}".format(exc))
    scene.cycles.device = "CPU"
    return "CPU"


def create_camera(scene, fov_radians):
    data = bpy.data.cameras.new("RadiosityGS_Camera")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.angle = fov_radians
    data.lens_unit = "FOV"
    data.clip_start = 0.01
    data.clip_end = 100.0
    camera = bpy.data.objects.new("RadiosityGS_Camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def create_point_light(scene, power, color):
    data = bpy.data.lights.new("RadiosityGS_Point", type="POINT")
    data.energy = power
    data.color = color
    data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("RadiosityGS_Point", data)
    scene.collection.objects.link(light)
    return light


def ring_positions(count, elevations, radius, phase, jitter, seed):
    """Deterministic complete-azimuth sampling on multiple elevation rings."""
    rng = random.Random(seed)
    ring_counts = [count // len(elevations)] * len(elevations)
    for i in range(count % len(elevations)):
        ring_counts[i] += 1
    positions = []
    for ring, (elevation, ring_count) in enumerate(zip(elevations, ring_counts)):
        elevation = math.radians(elevation)
        ring_phase = phase + ring * (math.pi * (3.0 - math.sqrt(5.0)))
        for i in range(ring_count):
            azimuth = ring_phase + 2.0 * math.pi * (i + 0.5) / ring_count
            r = radius * (1.0 + rng.uniform(-jitter, jitter))
            positions.append(Vector((
                r * math.cos(elevation) * math.cos(azimuth),
                r * math.cos(elevation) * math.sin(azimuth),
                r * math.sin(elevation),
            )))
    # Avoid grouping frames by elevation while retaining deterministic coverage.
    rng.shuffle(positions)
    return positions


def object_bounds(objects):
    lo = Vector((math.inf, math.inf, math.inf))
    hi = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
    if any(not math.isfinite(v) for v in (*lo, *hi)):
        raise RuntimeError("cannot compute object bounds")
    return lo, hi


def cornell_camera_positions(count, room_lo, room_hi, target, front_direction,
                             seed, phase=0.0):
    """Mix overview shots through the opening with rear/side interior views."""
    rng = random.Random(seed)
    front = Vector((front_direction.x, front_direction.y, 0.0)).normalized()
    side = Vector((-front.y, front.x, 0.0))
    room_center = (room_lo + room_hi) * 0.5
    horizontal_span = max(room_hi.x - room_lo.x, room_hi.y - room_lo.y)

    interior_count = 0 if count < 3 else max(2, int(round(count * 0.45)))
    front_count = count - interior_count
    positions = []

    # Views through the open face establish the whole room and front surfaces.
    for i in range(front_count):
        t = (i + 0.5 + phase) / max(front_count, 1)
        lateral = 0.30 * horizontal_span * math.sin(2.0 * math.pi * t)
        height = target.z + 0.18 + 0.20 * math.sin(4.0 * math.pi * t + 0.4)
        distance = 1.05 + rng.uniform(-0.10, 0.10)
        position = room_center + front * distance + side * lateral
        position.z = min(room_hi.z - 0.08, max(room_lo.z + 0.12, height))
        positions.append(position)

    # Cameras near the back and side walls look back across the inner boxes.
    # Keeping them above the box tops avoids placing cameras inside the props.
    inset = 0.09
    z_low = min(room_hi.z - 0.12, target.z + 0.30)
    z_high = room_hi.z - 0.10
    for i in range(interior_count):
        t = (i + 0.5 + phase) / interior_count
        wall = i % 3
        if wall == 0:  # rear wall, directly behind the boxes
            position = room_center - front * (0.5 * horizontal_span - inset)
            position += side * (0.26 * horizontal_span * math.sin(2.0 * math.pi * t))
        else:  # alternating side walls
            sign = -1.0 if wall == 1 else 1.0
            position = room_center + side * sign * (0.5 * horizontal_span - inset)
            position += front * (0.18 * horizontal_span * math.sin(2.0 * math.pi * t))
        position.z = rng.uniform(z_low, z_high)
        positions.append(position)
    rng.shuffle(positions)
    return positions


def cornell_light_positions(count, room_lo, room_hi, box_hi_z, seed):
    """Place known point lights inside the room, safely above both boxes."""
    rng = random.Random(seed)
    inset = 0.12
    z_low = min(room_hi.z - inset, max(box_hi_z + 0.10, room_lo.z + 0.58))
    z_high = room_hi.z - inset
    positions = []
    for _ in range(count):
        positions.append(Vector((
            rng.uniform(room_lo.x + inset, room_hi.x - inset),
            rng.uniform(room_lo.y + inset, room_hi.y - inset),
            rng.uniform(z_low, z_high),
        )))
    return positions


def aim_camera(camera, position, target=Vector((0.0, 0.0, 0.0))):
    camera.location = position
    camera.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def matrix_to_list(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def clear_known_outputs(output):
    for directory in (output / "train", output / "test"):
        if directory.exists():
            shutil.rmtree(str(directory))
    for filename in ("transforms_train.json", "transforms_test.json",
                     "dataset_info.json", "points3d.ply"):
        path = output / filename
        if path.exists():
            path.unlink()


def prepare_output(args):
    output = args.output.expanduser().resolve()
    known = [output / "train", output / "test", output / "transforms_train.json",
             output / "transforms_test.json", output / "dataset_info.json",
             output / "points3d.ply"]
    if any(path.exists() for path in known):
        if not args.overwrite:
            raise RuntimeError("dataset outputs already exist in {}; pass --overwrite".format(output))
        clear_known_outputs(output)
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "test").mkdir(parents=True, exist_ok=True)
    return output


def write_random_ply(path, count, lo, hi, seed):
    """Write loader initialization points, not ground-truth mesh samples."""
    rng = random.Random(seed)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Random volume initialization; no ground-truth geometry\n"
        "element vertex {}\n".format(count)
        + "property float x\nproperty float y\nproperty float z\n"
        + "property float nx\nproperty float ny\nproperty float nz\n"
        + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    record = struct.Struct("<ffffffBBB")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for _ in range(count):
            xyz = [rng.uniform(lo[i], hi[i]) for i in range(3)]
            handle.write(record.pack(xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, 25, 25, 25))


def render_split(scene, camera, light, output, split, camera_positions,
                 light_positions, args, phase_seed, aim_target=Vector((0.0, 0.0, 0.0))):
    frames = []
    rgb_power = [args.light_power * value for value in args.light_color]
    for index, (camera_pos, light_pos) in enumerate(zip(camera_positions, light_positions)):
        aim_camera(camera, camera_pos, aim_target)
        light.location = light_pos
        scene.cycles.seed = args.seed + phase_seed + index
        stem = "r_{:04d}".format(index)
        scene.render.filepath = str(output / split / (stem + ".exr"))
        print("[RadiosityGS] rendering {}/{} ({}/{})".format(
            split, stem, index + 1, len(camera_positions)))
        bpy.ops.render.render(write_still=True)
        frames.append({
            "file_path": "./{}/{}".format(split, stem),
            "transform_matrix": matrix_to_list(camera.matrix_world),
            "pl_pos": [float(value) for value in light_pos],
            "pl_intensity": rgb_power,
        })
    return {"camera_angle_x": math.radians(args.fov), "frames": frames}


def main():
    args = parse_args()
    scene = bpy.context.scene
    source_camera_position = None
    if scene.camera is not None:
        source_camera_position = scene.camera.matrix_world.translation.copy()
    cornell_objects = [obj for obj in scene.objects
                       if obj.type in GEOMETRY_TYPES and "cornell" in obj.name.lower()]
    box_objects = [obj for obj in scene.objects
                   if obj.type in GEOMETRY_TYPES and "box" in obj.name.lower()
                   and obj not in cornell_objects]
    layout = args.layout
    if layout == "auto":
        layout = "cornell" if cornell_objects and box_objects else "exterior"
    if args.light_power is None:
        args.light_power = 10.0 if layout == "cornell" else 100.0

    original_lo, original_hi, object_count = evaluated_bounds(scene)
    remove_existing_cameras_and_lights()
    center, scale, normalization = normalize_scene(scene, original_lo, original_hi, args.margin)
    emission_materials = disable_unmodelled_emission(scene)
    normalized_lo, normalized_hi, _ = evaluated_bounds(scene)

    extent = normalized_hi - normalized_lo
    bound_radius = 0.5 * extent.length
    half_fov = 0.5 * math.radians(args.fov)
    camera_radius = bound_radius / math.sin(half_fov) * 1.15
    # Keep every light at least one unit from the bounding sphere. RadiosityGS's
    # default inverse-square falloff clamp then agrees with Cycles everywhere.
    light_radius = bound_radius + 1.05

    print("[RadiosityGS] {} evaluated geometry instances".format(object_count))
    print("[RadiosityGS] original bounds: {} .. {}".format(tuple(original_lo), tuple(original_hi)))
    print("[RadiosityGS] normalization: center={}, scale={:.9g}".format(tuple(center), scale))
    print("[RadiosityGS] normalized bounds: {} .. {}".format(tuple(normalized_lo), tuple(normalized_hi)))
    print("[RadiosityGS] camera radius={:.4f}, light radius={:.4f}".format(
        camera_radius, light_radius))
    print("[RadiosityGS] camera layout={}".format(layout))
    if emission_materials:
        print("[RadiosityGS] disabled unmodelled emission in: {}".format(", ".join(emission_materials)))
    if args.dry_run:
        print("[RadiosityGS] dry run complete; no files written")
        return

    output = prepare_output(args)
    render_device = configure_cycles(scene, args)
    camera = create_camera(scene, math.radians(args.fov))
    light = create_point_light(scene, args.light_power, tuple(args.light_color))

    aim_target = Vector((0.0, 0.0, 0.0))
    recommended_falloff = 1.0
    if layout == "cornell":
        if not cornell_objects or not box_objects:
            raise RuntimeError("--layout cornell needs geometry named like 'cornell' and 'box'")
        room_lo, room_hi = object_bounds(cornell_objects)
        boxes_lo, boxes_hi = object_bounds(box_objects)
        aim_target = (boxes_lo + boxes_hi) * 0.5
        room_center = (room_lo + room_hi) * 0.5
        if source_camera_position is not None:
            normalized_source_camera = normalization @ source_camera_position
            front_direction = normalized_source_camera - room_center
            front_direction.z = 0.0
        else:
            front_direction = Vector((0.0, -1.0, 0.0))
        if front_direction.length_squared < 1e-8:
            front_direction = Vector((0.0, -1.0, 0.0))
        train_cameras = cornell_camera_positions(
            args.train_views, room_lo, room_hi, aim_target, front_direction,
            args.seed + 1)
        test_cameras = cornell_camera_positions(
            args.test_views, room_lo, room_hi, aim_target, front_direction,
            args.seed + 2, phase=0.5)
        train_lights = cornell_light_positions(
            args.train_views, room_lo, room_hi, boxes_hi.z, args.seed + 3)
        test_lights = cornell_light_positions(
            args.test_views, room_lo, room_hi, boxes_hi.z, args.seed + 4)
        # Interior lights can be closer than one scene unit. Do not use the
        # trainer's default inverse-square clamp for this layout.
        recommended_falloff = 100.0
    else:
        train_cameras = ring_positions(args.train_views, args.camera_elevations,
                                       camera_radius, 0.0, 0.04, args.seed + 1)
        test_cameras = ring_positions(args.test_views, args.camera_elevations,
                                      camera_radius, math.pi / args.test_views, 0.02, args.seed + 2)
        train_lights = ring_positions(args.train_views, args.light_elevations,
                                      light_radius, 0.37, 0.08, args.seed + 3)
        test_lights = ring_positions(args.test_views, args.light_elevations,
                                     light_radius, 0.37 + math.pi / args.test_views,
                                     0.04, args.seed + 4)

    train_json = render_split(scene, camera, light, output, "train",
                              train_cameras, train_lights, args, 10_000, aim_target)
    test_json = render_split(scene, camera, light, output, "test",
                             test_cameras, test_lights, args, 20_000, aim_target)
    for split, contents in (("train", train_json), ("test", test_json)):
        with (output / ("transforms_{}.json".format(split))).open("w", encoding="utf-8") as handle:
            json.dump(contents, handle, indent=2)
            handle.write("\n")

    if args.initial_points:
        write_random_ply(output / "points3d.ply", args.initial_points,
                         normalized_lo, normalized_hi, args.seed + 5)
    info = {
        "format": "RadiosityGS / GS^3-style Blender",
        "source_blend": bpy.data.filepath,
        "linear_hdr": True,
        "resolution": [args.resolution, args.resolution],
        "samples": args.samples,
        "train_views": args.train_views,
        "test_views": args.test_views,
        "layout": layout,
        "camera_elevations_deg": args.camera_elevations,
        "light_elevations_deg": args.light_elevations,
        "camera_radius": camera_radius,
        "light_radius": light_radius,
        "light_power_rgb": [args.light_power * value for value in args.light_color],
        "aim_target": list(aim_target),
        "recommended_max_inverse_falloff": recommended_falloff,
        "original_bounds": [list(original_lo), list(original_hi)],
        "normalized_bounds": [list(normalized_lo), list(normalized_hi)],
        "normalization_center": list(center),
        "normalization_scale": scale,
        "disabled_emissive_materials": emission_materials,
        "cycles_device": render_device,
        "seed": args.seed,
    }
    with (output / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
    print("[RadiosityGS] complete: {}".format(output))


if __name__ == "__main__":
    main()
