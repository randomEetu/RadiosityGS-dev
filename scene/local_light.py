"""Shared tensor interface for homogeneous local-light rigs.

``renderGI`` represents every local emitter as a tiny Gaussian surfel. Point
and spot lights therefore share the same geometry; only their emission SH
coefficients and, for spotlights, rotation differ.
"""

import torch


LOCAL_LIGHT_SCALE = (1e-3, 1e-3)
LOCAL_LIGHT_GEOVALUE = 6.0
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def point_emissions(dc: torch.Tensor, max_sh_degree: int) -> torch.Tensor:
    """Pack ``(N, 3)`` SH-DC coefficients into an isotropic emission tensor."""
    dc = dc[:, None, :]
    coefficient_count = (max_sh_degree + 1) ** 2
    higher_bands = torch.zeros(
        (len(dc), coefficient_count - 1, 3), dtype=dc.dtype, device=dc.device
    )
    return torch.cat((dc, higher_bands), dim=1)


class LocalLightGeometry:
    """Fixed Gaussian-surfel geometry needed by a local light source.

    Subclasses provide ``self._xyz`` with shape ``(N, 3)``. Spotlights override
    ``get_rotation``; directional SH lights set ``clamp_emissions = False``.
    """

    is_directional_light = False
    clamp_emissions = True

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.tensor(LOCAL_LIGHT_SCALE, device=self._device)[None].repeat(len(self._xyz), 1)

    @property
    def get_rotation(self):
        return torch.tensor(IDENTITY_QUATERNION, device=self._device)[None].repeat(len(self._xyz), 1)

    @property
    def get_geovalue(self):
        return torch.full((len(self._xyz), 1), LOCAL_LIGHT_GEOVALUE, device=self._device)

    @property
    def get_norm_factor(self):
        return torch.ones((len(self._xyz), 1), device=self._device)

    @property
    def get_is_light_source(self):
        return torch.ones((len(self._xyz), 1), dtype=torch.bool, device=self._device)
