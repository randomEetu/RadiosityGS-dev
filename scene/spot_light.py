#
# scene/spot_light.py
#
# A SPOTLIGHT for RadiosityGS, implemented *entirely* on the Python side -- no
# CUDA change and no recompilation.
#
# How it works
# ------------
# The solver already treats a light source as a special Gaussian surfel whose
# outgoing radiance is an SH function evaluated in the light's OWN LOCAL FRAME:
#
#   one_bounce_estimator/src/optix_dev.cu:173
#       ray_dir_emit_frame = ray_dir * emit_rotation_matrix    // = R^T * (recv - emit)
#       c = computeOutRadianceFromSH(deg, emissions + emit_idx * STRIDE, ray_dir_emit_frame, clamp)
#
# A point light just leaves every band but the DC one at zero, which makes `c`
# isotropic. So to get a spotlight we only have to
#
#   1. aim the light: pick the rotation quaternion whose local +z axis is the
#      spot direction (the emission profile is symmetric around +z, so the other
#      two axes are arbitrary), and
#   2. put the angular attenuation profile into the light's SH emission bands.
#
# The profile is exactly the one we want to attenuate with:
#
#       factor(theta) = 1                                   if theta <= cutoff
#                     = exp(-(theta - cutoff)^2 / 2 sigma^2) otherwise
#
# (the Gaussian is centred at the cutoff, so it starts at 1 and decays to 0 --
# it is also C^1 there, which keeps the SH fit well behaved).
#
# Caveats of encoding the profile in SH -- read these
# ---------------------------------------------------
# * BAND-LIMITED. The scene's SH degree (9) resolves angular features of roughly
#   180/9 = 20 degrees. A very tight spot with a crisp edge cannot be
#   represented; ask for `cutoff + sigma` of at least ~15-20 degrees. `fit_report`
#   tells you how badly the fit misses, so you can check instead of guess.
# * TWO-SIDED. `computeOutRadianceFromSH` folds the direction into the z >= 0
#   hemisphere ("Convention: Always pointing outwards", auxiliary.h:82), so the
#   emission is necessarily symmetric under d -> -d: every spotlight is really a
#   double-ended cone, with a mirror lobe pointing backwards. Aim the light so
#   the back lobe shoots into empty space and you will never see it.
# * RINGING. The band-limited profile overshoots slightly outside the cone.
#   Undershoots are harmless (the solver clamps the evaluated radiance to >= 0 in
#   the direct pass); overshoots show up as a faint halo of a few percent.
# * The 'MC' solver's next-event importance sampling weights emitters by their DC
#   band alone (next_event.cu:112 evaluates the emission at degree 0), which for a
#   narrow lobe is a poor match for its actual radiance. The estimator divides by
#   that pdf, so the result stays unbiased -- just noisier; raise --num_walks if
#   you use 'MC'. The default 'hybrid' solver is unaffected: it handles the light
#   in the progressive-refinement pass and zeroes it before the MC rounds.
#
# Everything else -- shadows, 1/r^2 falloff, the indirect bounces -- is handled
# by the untouched solver, because the spotlight is still just a light source
# primitive to it.
#

import math

import torch

from utils.sh_utils import eval_sh, eval_sh_response


def spot_factor(angle_rad: torch.Tensor, cutoff_deg: float, sigma_deg: float) -> torch.Tensor:
    """The attenuation profile: flat 1.0 inside the cone, Gaussian roll-off outside.

    ``angle_rad`` is the angle between the spot axis and the direction from the
    light towards the receiver.
    """
    cutoff = math.radians(cutoff_deg)
    sigma = math.radians(max(sigma_deg, 1e-6))
    excess = (angle_rad - cutoff).clamp_min(0.)
    return torch.exp(-0.5 * (excess / sigma) ** 2)


def fibonacci_hemisphere(num_samples: int, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """``num_samples`` roughly equal-area directions on the z >= 0 hemisphere.

    Equal area matters: the SH fit below is a least-squares problem, and equal
    area sample weights make it the L2 projection onto the hemisphere.
    """
    i = torch.arange(num_samples, device=device, dtype=dtype) + 0.5
    z = 1. - i / num_samples                       # uniform in z  =>  equal area
    r = torch.sqrt((1. - z * z).clamp_min(0.))
    phi = i * math.pi * (3. - math.sqrt(5.))       # golden angle
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi), z), dim=-1)


def _band_of_coefficient(sh_degree: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """``l`` for every coefficient index, i.e. [0, 1,1,1, 2,2,2,2,2, ...]."""
    return torch.tensor([l for l in range(sh_degree + 1) for _ in range(2 * l + 1)],
                        dtype=dtype, device=device)


def fit_spot_sh(
    sh_degree: int,
    cutoff_deg: float,
    sigma_deg: float,
    num_samples: int = 20000,
    ridge: float = 1e-8,
    device="cpu",
):
    """Least-squares fit of ``spot_factor`` onto the renderer's SH basis.

    Returns ``(coeffs, report)``: ``coeffs`` is a ``((sh_degree+1)^2,)`` tensor of
    SH coefficients such that ``eval_sh(sh_degree, coeffs, d) ~= spot_factor(angle
    between d and +z)``, i.e. a *scalar* profile with a peak of 1.

    The design matrix comes from ``eval_sh_response``, which is the same generated
    code as the CUDA ``computeOutRadianceFromSH`` (both fold the direction into
    z >= 0), so the fit is expressed in the renderer's own convention -- no
    normalisation assumptions of our own. Consequence: the fit only has to hold on
    a HEMISPHERE, which buys noticeably more angular resolution than projecting an
    antipodally symmetric profile over the whole sphere would.

    ``ridge`` weights a Laplacian-style smoothness penalty (``(l(l+1))^2``). The
    degree-9 basis restricted to a hemisphere is linearly independent but poorly
    conditioned, so without it the fit is free to pick huge cancelling
    coefficients: for a 10 deg cone the peak coefficient goes from ~2e0 at
    ridge=1e-2 to ~3e4 at ridge=0, and float32 evaluation starts to bite. 1e-8
    keeps |c| ~< 1e3 (float32 error < 1e-3 of the peak) while barely blunting the
    fit.
    """
    dirs = fibonacci_hemisphere(num_samples, device=device, dtype=torch.float64)
    basis = eval_sh_response(sh_degree, 1., dirs).reshape(len(dirs), -1)  # (M, K)
    target = spot_factor(torch.acos(dirs[:, 2].clamp(-1., 1.)), cutoff_deg, sigma_deg)

    l = _band_of_coefficient(sh_degree, dtype=basis.dtype, device=device)
    penalty = (l * (l + 1)) ** 2
    gram = basis.T @ basis
    gram = gram + ridge * (torch.trace(gram) / len(gram)) * torch.diag(penalty / penalty.clamp_min(1.).max())
    coeffs = torch.linalg.solve(gram, basis.T @ target)

    return coeffs.float(), fit_report(coeffs, sh_degree, cutoff_deg, sigma_deg, device=device)


@torch.no_grad()
def fit_report(coeffs: torch.Tensor, sh_degree: int, cutoff_deg: float, sigma_deg: float,
               device="cpu", num_angles: int = 1441):
    """Compare the fitted profile against the requested one on a dense angle grid."""
    theta = torch.linspace(0., math.pi, num_angles, device=device, dtype=torch.float64)
    dirs = torch.stack((torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)), dim=-1)
    # eval_sh folds d -> -d for z < 0 exactly like the CUDA does, so this grid also
    # shows the unavoidable mirror lobe on the theta > 90 deg side.
    # eval_sh wants (..., C, K) coefficients against (..., 3) directions; one
    # "channel" is enough since the profile is scalar.
    def evaluate(c, d):
        return eval_sh(sh_degree, c.reshape(1, 1, -1).expand(num_angles, 1, -1), d).reshape(-1)

    fitted = evaluate(coeffs.to(dirs.dtype), dirs)
    wanted = spot_factor(theta, cutoff_deg, sigma_deg)
    front = theta <= math.pi / 2
    # "Outside" = past the cone but not yet into the mirror lobe.
    edge = math.radians(min(cutoff_deg + 3. * sigma_deg, 89.))
    outside = (theta > edge) & (theta < math.pi - edge)
    half_power = theta[fitted >= 0.5 * max(fitted[0].item(), 1e-12)]
    return {
        "degree": sh_degree,
        "max_abs_error": (fitted[front] - wanted[front]).abs().max().item(),
        "peak": fitted.max().item(),
        "on_axis": fitted[0].item(),
        # eval_sh clamps at 0 like the renderer does in the direct pass, so
        # undershoot is harmless; overshoot is the visible halo outside the cone.
        "halo": fitted[outside].max().item() if outside.any() else 0.,
        # Where the *actual* (band-limited) cone edge ends up.
        "half_power_deg": math.degrees(half_power[half_power <= math.pi / 2].max().item())
        if (half_power <= math.pi / 2).any() else 0.,
        "float32_error": (fitted - evaluate(coeffs.float().to(torch.float64), dirs)).abs().max().item(),
        "theta_deg": theta * 180. / math.pi,
        "fitted": fitted,
        "wanted": wanted,
    }


def ascii_profile(report, width: int = 62, rows: int = 14) -> str:
    """A terminal plot of requested (``.``) vs fitted (``#``) profile, 0..180 deg."""
    theta, fitted, wanted = report["theta_deg"], report["fitted"], report["wanted"]
    idx = torch.linspace(0, len(theta) - 1, width).long()
    hi = max(1.0, report["peak"])
    lines = []
    for r in range(rows):
        lo_v, hi_v = hi * (rows - 1 - r) / rows, hi * (rows - r) / rows
        row = "".join(
            "#" if lo_v <= fitted[i] < hi_v or (r == 0 and fitted[i] >= lo_v)
            else ("." if lo_v <= wanted[i] < hi_v or (r == 0 and wanted[i] >= lo_v) else " ")
            for i in idx)
        lines.append(f"{hi_v:5.2f} |{row}")
    lines.append("      +" + "-" * width)
    ticks = "".join(f"{int(theta[i].item()):<{width // 6}d}" for i in idx[::width // 6][:6])
    lines.append("       " + ticks + "  [deg from spot axis]")
    return "\n".join(lines)


def quat_from_z_to(direction: torch.Tensor) -> torch.Tensor:
    """Quaternion ``(w, x, y, z)`` rotating local +z onto ``direction``.

    Matches ``build_rotation`` in the solver / rasterizer (same ``(w, x, y, z)``
    ordering), so ``R(q)[:, 2] == direction`` and therefore the SH lobe we build
    around local +z ends up pointing along ``direction`` in world space.
    """
    d = torch.nn.functional.normalize(direction.reshape(3).double(), dim=0)
    axis = torch.stack((-d[1], d[0], torch.zeros_like(d[0])))  # z_hat x d
    sin_a, cos_a = axis.norm(), d[2]
    if sin_a < 1e-8:
        # Parallel or antiparallel to +z.
        q = torch.tensor([1., 0., 0., 0.] if cos_a > 0 else [0., 1., 0., 0.], dtype=d.dtype, device=d.device)
    else:
        angle = torch.atan2(sin_a, cos_a)
        q = torch.cat((torch.cos(angle / 2).reshape(1), axis / sin_a * torch.sin(angle / 2)))
    return (q / q.norm()).float()


class SpotLight:
    """A single point light with a cone-shaped emission profile.

    Duck-types the light-source object ``renderGI`` expects (the same interface as
    ``LightModel.from_camera_if_possible`` / ``LearnablePointLight``). Not
    trainable: this is for relighting a frozen scene from arbitrary spot
    positions and angles.

    ``intensity`` is linear RGB radiance at the centre of the cone, in the same
    units as a dataset point light's ``pl_intensity`` before ``RGB2SH``, so
    passing the dataset's mean RGB reproduces its brightness on the axis.
    """

    # renderGI must not clamp our emission coefficients to >= 0: the higher bands
    # are genuinely negative, that is what shapes the lobe.
    clamp_emissions = False

    def __init__(self, position, direction, intensity, cutoff_deg=25., sigma_deg=15.,
                 max_sh_degree=9, fit_sh_degree=None, num_samples=20000, ridge=1e-8, device="cuda"):
        self.max_sh_degree = max_sh_degree
        self.fit_sh_degree = max_sh_degree if fit_sh_degree is None else fit_sh_degree
        assert 0 <= self.fit_sh_degree <= max_sh_degree
        self.is_directional_light = False
        self.cutoff_deg = cutoff_deg
        self.sigma_deg = sigma_deg
        self._device = device

        self._xyz = torch.as_tensor(position, dtype=torch.float32, device=device).reshape(1, 3)
        self._direction = torch.nn.functional.normalize(
            torch.as_tensor(direction, dtype=torch.float32, device=device).reshape(3), dim=0)
        self._intensity = torch.as_tensor(intensity, dtype=torch.float32, device=device).reshape(3)

        # The profile fit only depends on (degree, cutoff, sigma) -- do it on the
        # CPU in double precision, once.
        profile, self.fit = fit_spot_sh(self.fit_sh_degree, cutoff_deg, sigma_deg,
                                        num_samples=num_samples, ridge=ridge)
        n_bands = (max_sh_degree + 1) ** 2
        self._profile = torch.zeros(n_bands, dtype=torch.float32, device=device)
        self._profile[: len(profile)] = profile.to(device)

    # --- aiming ---
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return quat_from_z_to(self._direction)[None].to(self._device)

    def look_at(self, target):
        """Aim the cone at a world-space point."""
        target = torch.as_tensor(target, dtype=torch.float32, device=self._device).reshape(3)
        self._direction = torch.nn.functional.normalize(target - self._xyz.reshape(3), dim=0)
        return self

    def place(self, position, target=None):
        """Move the light, optionally re-aiming it at ``target``."""
        self._xyz = torch.as_tensor(position, dtype=torch.float32, device=self._device).reshape(1, 3).contiguous()
        return self.look_at(target) if target is not None else self

    # --- emission ---
    @property
    def get_emissions(self):
        # (1, (deg+1)^2, 3). The scalar profile has a peak of 1, so scaling it by
        # the RGB radiance gives `spot_factor(theta) * intensity` on evaluation.
        return (self._profile[None, :, None] * self._intensity[None, None, :]).contiguous()

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
    def get_is_light_source(self):
        return torch.tensor([True], device=self._device)[None]

    def describe(self):
        pos = self._xyz.reshape(3).tolist()
        d = self._direction.tolist()
        rgb = self._intensity.tolist()
        return (f"spotlight pos [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                f"dir [{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}] "
                f"RGB [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}] "
                f"cutoff {self.cutoff_deg:g} deg, sigma {self.sigma_deg:g} deg, SH degree {self.fit_sh_degree}\n"
                f"           SH fit: max err {self.fit['max_abs_error']:.3f}, on-axis {self.fit['on_axis']:.3f}, "
                f"half-power at {self.fit['half_power_deg']:.1f} deg, halo {self.fit['halo']:.3f}, "
                f"fp32 err {self.fit['float32_error']:.1e}")
