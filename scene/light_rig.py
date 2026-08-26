"""Serialization and loading for homogeneous local-light rigs."""

import json

import torch

from scene.light_source import PointLights
from scene.spot_light import SpotLight


def _read(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("lights"), list):
        raise ValueError("light rig must be an object containing a 'lights' list")
    if not data["lights"]:
        raise ValueError("light rig must contain at least one light")

    declared = data.get("type")
    entry_types = {entry.get("type", declared) for entry in data["lights"]}
    if None in entry_types:
        raise ValueError("light rig must declare its type ('point' or 'spot')")
    if len(entry_types) != 1:
        raise ValueError("mixed point/spot rigs are not supported")
    light_type = entry_types.pop()
    if declared is not None and declared != light_type:
        raise ValueError("top-level light type disagrees with a light entry")
    if light_type not in ("point", "spot"):
        raise ValueError(f"unsupported light type: {light_type!r}")
    return light_type, data["lights"]


def load_light_rig(path, max_sh_degree=9, fit_sh_degree=None, device="cuda"):
    """Load a fixed point- or spotlight batch from a JSON file."""
    light_type, lights = _read(path)
    positions = [entry["position"] for entry in lights]
    intensities = [entry["intensity"] for entry in lights]
    if light_type == "point":
        return PointLights(positions, intensities, max_sh_degree=max_sh_degree, device=device)

    directions = []
    for entry, position in zip(lights, positions):
        if "direction" in entry:
            directions.append(entry["direction"])
        elif "target" in entry:
            directions.append((torch.tensor(entry["target"]) - torch.tensor(position)).tolist())
        else:
            raise ValueError("each spotlight needs either 'direction' or 'target'")
    return SpotLight(
        positions, directions, intensities,
        cutoff_deg=[entry.get("cutoff_deg", 20.) for entry in lights],
        sigma_deg=[entry.get("sigma_deg", 12.) for entry in lights],
        max_sh_degree=max_sh_degree,
        fit_sh_degree=fit_sh_degree,
        ridge=[entry.get("ridge", 1e-8) for entry in lights][0],
        device=device,
    )


def save_light_rig(path, light_type, positions, intensities, directions=None,
                   cutoff_deg=None, sigma_deg=None):
    """Write a human-readable rig. Intensities are linear RGB."""
    positions = torch.as_tensor(positions).detach().cpu().reshape(-1, 3)
    intensities = torch.as_tensor(intensities).detach().cpu().reshape(-1, 3)
    if positions.shape != intensities.shape:
        raise ValueError("positions and intensities must both have shape (N, 3)")
    if light_type not in ("point", "spot"):
        raise ValueError("light_type must be 'point' or 'spot'")

    entries = []
    if light_type == "spot":
        directions = torch.as_tensor(directions).detach().cpu().reshape(-1, 3)
        if directions.shape != positions.shape:
            raise ValueError("spotlight directions must have shape (N, 3)")
        cutoff_deg = _broadcast(cutoff_deg, len(positions), "cutoff_deg")
        sigma_deg = _broadcast(sigma_deg, len(positions), "sigma_deg")

    for i in range(len(positions)):
        entry = {
            "position": positions[i].tolist(),
            "intensity": intensities[i].tolist(),
        }
        if light_type == "spot":
            entry.update({
                "direction": directions[i].tolist(),
                "cutoff_deg": cutoff_deg[i],
                "sigma_deg": sigma_deg[i],
            })
        entries.append(entry)
    with open(path, "w") as f:
        json.dump({"type": light_type, "lights": entries}, f, indent=2)
        f.write("\n")


def _broadcast(value, count, name):
    values = torch.as_tensor(value, dtype=torch.float64).reshape(-1).tolist()
    if len(values) == 1:
        values *= count
    if len(values) != count:
        raise ValueError(f"{name} must be scalar or contain one value per light")
    return values
