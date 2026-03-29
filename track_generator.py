"""
track_generator.py — Procedural closed-circuit track generator for F1TENTH.
============================================================================
Generates random closed-loop race tracks compatible with f110_gym, producing:
  - Occupancy-grid PNG image  (white = free, black = wall)
  - Map metadata YAML         (ROS map_server format)
  - Centerline CSV             (x_m, y_m, w_tr_right_m, w_tr_left_m)
  - Raceline CSV               (s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2)

Designed to be called between training generations to expand the training
distribution with novel track layouts.

Usage:
    gen = TrackGenerator(min_track_length=200, max_track_length=800)
    map_dir = gen.generate("my_track")          # → "maps/generated/my_track"
    map_dir = gen.generate()                     # auto-named

Integration with training loop:
    gen = TrackGenerator()
    for generation in range(num_generations):
        name = f"gen_{generation}"
        gen.generate(name)
        env.update_map(get_map_dir(name) + f"/{name}_map", ".png")
        # ... run generation ...

The generated map directory is placed directly under maps/ (NOT a
subdirectory) so that get_map_dir(name) works unchanged.
"""

import os
import uuid
import math

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw


class TrackGenerator:
    """
    Procedural F1TENTH track generator.

    Generates random closed tracks by:
      1. Placing control points on a randomly deformed ring
      2. Fitting a periodic cubic B-spline through them
      3. Enforcing curvature constraints via iterative smoothing
      4. Computing heading, curvature, arc-length, and track widths
      5. Rendering an occupancy-grid image and writing all map files
    """

    def __init__(
        self,
        output_base_dir: str = "maps",
        # ── Track length ──────────────────────────────────────────
        min_track_length: float = 100.0,      # meters
        max_track_length: float = 2000.0,     # meters
        # ── Track width ───────────────────────────────────────────
        min_track_width: float = 1.0,         # meters (total, each side ≈ half)
        max_track_width: float = 2.5,         # meters
        variable_width: bool = True,          # widen on straights, narrow in turns
        # ── Turns ─────────────────────────────────────────────────
        min_turns: int = 6,                   # minimum control-point turns
        max_turns: int = 35,
        min_turn_radius: float = 3.5,         # meters (limits max curvature)
        max_turn_radius: float = 50.0,        # meters (informational)
        # ── Ring perturbation ─────────────────────────────────────
        radius_noise: float = 0.5,            # relative noise on ring radius
        angle_noise: float = 0.25,            # relative noise on angular spacing
        # ── Map rendering ─────────────────────────────────────────
        resolution: float = 0.05,             # meters per pixel
        margin_meters: float = 5.0,           # empty border around track
        max_image_dim: int = 8000,            # auto-coarsen if exceeded
        # ── Raceline / centerline output ──────────────────────────
        raceline_speed: float = 8.0,          # constant vx_mps in raceline CSV
        raceline_ds: float = 0.2,             # arc-length step for raceline CSV
        centerline_ds: float = 0.4,           # arc-length step for centerline CSV
        # ── Generation control ────────────────────────────────────
        max_attempts: int = 50,
        seed: int | None = None,
    ):
        """
        Args:
            output_base_dir: Base directory for generated map folders.
            min_track_length: Minimum track centerline length (meters).
            max_track_length: Maximum track centerline length (meters).
            min_track_width: Minimum total track width (meters).
            max_track_width: Maximum total track width (meters).
            variable_width:  Widen on straights, narrow in turns.
            min_turns: Minimum number of control-point turns (≥6).
            max_turns: Maximum number of control-point turns.
            min_turn_radius: Minimum turn radius (meters). Max curvature = 1/this.
            max_turn_radius: (informational) Maximum expected turn radius.
            radius_noise: Controls shape variation (0 = circle, >0 = deformed).
            angle_noise: Controls angular spacing variation.
            resolution: Meters per pixel in the occupancy image.
            margin_meters: Empty border around track in the image (meters).
            max_image_dim: Maximum image dimension; auto-coarsens if exceeded.
            raceline_speed: Constant vx_mps written to raceline CSV.
            raceline_ds: Arc-length step between raceline CSV points (meters).
            centerline_ds: Arc-length step between centerline CSV points (meters).
            max_attempts: Maximum generation attempts before error.
            seed: Random seed for reproducibility. None = random.
        """
        self.output_base_dir = output_base_dir
        self.min_track_length = min_track_length
        self.max_track_length = max_track_length
        self.min_track_width = min_track_width
        self.max_track_width = max_track_width
        self.variable_width = variable_width
        self.min_turns = max(min_turns, 6)
        self.max_turns = max(max_turns, self.min_turns)
        self.min_turn_radius = min_turn_radius
        self.max_turn_radius = max_turn_radius
        self.radius_noise = radius_noise
        self.angle_noise = angle_noise
        self.resolution = resolution
        self.margin_meters = margin_meters
        self.max_image_dim = max_image_dim
        self.raceline_speed = raceline_speed
        self.raceline_ds = raceline_ds
        self.centerline_ds = centerline_ds
        self.max_attempts = max_attempts
        self.rng = np.random.default_rng(seed)

        # Boundary constraint: turn radius must exceed half track width
        if self.min_turn_radius < self.max_track_width / 2.0:
            self.min_turn_radius = self.max_track_width / 2.0 + 0.5

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def generate(self, name: str | None = None) -> str:
        """
        Generate a new closed-loop track and write all map files.

        Args:
            name: Track/directory name. Auto-generated if None.

        Returns:
            Track name (usable with ``get_map_dir(name)``).

        Raises:
            RuntimeError: If no valid track could be generated.
        """
        if name is None:
            name = f"gen_{uuid.uuid4().hex[:8]}"

        map_dir = os.path.join(self.output_base_dir, name)
        os.makedirs(map_dir, exist_ok=True)

        for _ in range(self.max_attempts):
            result = self._try_generate(name, map_dir)
            if result is not None:
                return result

        raise RuntimeError(
            f"Failed to generate a valid track after {self.max_attempts} attempts. "
            f"Try relaxing constraints (larger min_turn_radius, smaller max_track_width, etc.)."
        )

    # ─────────────────────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────────────────────

    def _try_generate(self, name, map_dir):
        """Single attempt. Returns *name* on success, None on failure."""
        try:
            # 1. Target track length for this attempt
            target_length = float(
                self.rng.uniform(self.min_track_length, self.max_track_length))

            # 2. Control points on a deformed ring
            ctrl = self._generate_control_points(target_length)

            # 3. Fit periodic spline
            tck, raw_length = self._fit_periodic_spline(ctrl)

            # 4. Scale to match target length
            scale = target_length / max(raw_length, 1e-6)
            ctrl = ctrl * scale
            tck, total_length = self._fit_periodic_spline(ctrl)

            # 5. Resample at fine resolution (~0.1 m)
            fine_ds = min(0.1, total_length / 500)
            fine_xy, fine_s, fine_psi, fine_kappa = self._resample_spline(
                tck, total_length, fine_ds)

            # 6. Enforce curvature constraints
            max_kappa = 1.0 / self.min_turn_radius
            if np.any(np.abs(fine_kappa) > max_kappa):
                fine_xy, fine_s, fine_psi, fine_kappa = self._smooth_curvature(
                    fine_xy, max_kappa)

            # Recompute total length after smoothing
            total_length = fine_s[-1] + np.linalg.norm(
                fine_xy[0] - fine_xy[-1])

            # 7. Length check
            if not (self.min_track_length * 0.7 <= total_length
                    <= self.max_track_length * 1.3):
                return None

            # 8. Track widths
            w_right, w_left = self._generate_widths(
                fine_kappa, len(fine_xy))

            # 9. Boundaries
            normals = self._compute_normals(fine_psi)
            left_bnd = fine_xy + normals * w_left[:, None]
            right_bnd = fine_xy - normals * w_right[:, None]

            # 10. Validate boundaries
            if not self._boundaries_valid(fine_xy, w_left + w_right):
                return None

            # 11. Downsample for output files
            rl_xy, rl_s, rl_psi, rl_kappa = self._downsample(
                fine_xy, fine_s, fine_psi, fine_kappa, self.raceline_ds)
            cl_xy, cl_s, cl_psi, cl_kappa = self._downsample(
                fine_xy, fine_s, fine_psi, fine_kappa, self.centerline_ds)
            cl_wr = np.interp(cl_s, fine_s, w_right)
            cl_wl = np.interp(cl_s, fine_s, w_left)

            # 12. Render occupancy image
            img, origin, eff_res = self._render_occupancy(
                left_bnd, right_bnd)

            # 13. Write all files
            self._write_image(
                os.path.join(map_dir, f"{name}_map.png"), img)
            self._write_yaml(
                os.path.join(map_dir, f"{name}_map.yaml"),
                f"{name}_map.png", eff_res, origin)
            self._write_raceline_csv(
                os.path.join(map_dir, f"{name}_raceline.csv"),
                rl_s, rl_xy, rl_psi, rl_kappa)
            self._write_centerline_csv(
                os.path.join(map_dir, f"{name}_centerline.csv"),
                cl_xy, cl_wr, cl_wl)

            return name

        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────
    # Control-point generation
    # ─────────────────────────────────────────────────────────────

    def _generate_control_points(self, target_length):
        """Place control points on a randomly deformed ring."""
        rng = self.rng
        n = int(rng.integers(self.min_turns, self.max_turns + 1))
        base_radius = target_length / (2.0 * math.pi)

        # Evenly-spaced base angles
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)

        # Perturb angles (maintain ordering)
        step = 2 * math.pi / n
        angles += rng.uniform(
            -self.angle_noise * step, self.angle_noise * step, size=n)
        angles = np.sort(angles % (2 * math.pi))

        # Perturb radii
        radii = base_radius * (
            1.0 + rng.uniform(-self.radius_noise, self.radius_noise, size=n))
        radii = np.maximum(radii, 2.0)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])

    # ─────────────────────────────────────────────────────────────
    # Spline fitting & resampling
    # ─────────────────────────────────────────────────────────────

    def _fit_periodic_spline(self, control_points):
        """Fit a periodic cubic spline. Returns (tck, total_arc_length)."""
        pts = np.vstack([control_points, control_points[:1]])
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)

        u_fine = np.linspace(0, 1, 10000, endpoint=False)
        x_f, y_f = splev(u_fine, tck)
        ds = np.sqrt(np.diff(x_f) ** 2 + np.diff(y_f) ** 2)
        return tck, float(np.sum(ds))

    def _resample_spline(self, tck, total_length, ds):
        """Resample periodic spline at uniform arc-length spacing.

        Returns (xy, s, psi, kappa).
        """
        n_fine = max(20000, int(total_length / 0.01))
        u_fine = np.linspace(0, 1, n_fine, endpoint=False)
        x_f, y_f = splev(u_fine, tck)

        seg_ds = np.sqrt(np.diff(x_f) ** 2 + np.diff(y_f) ** 2)
        s_fine = np.concatenate([[0.0], np.cumsum(seg_ds)])

        n_pts = max(int(total_length / ds), 50)
        s_uniform = np.linspace(0, total_length, n_pts, endpoint=False)
        u_of_s = np.interp(s_uniform, s_fine, u_fine)

        xx, yy = splev(u_of_s, tck)
        dx, dy = splev(u_of_s, tck, der=1)
        d2x, d2y = splev(u_of_s, tck, der=2)

        xy = np.column_stack([xx, yy])
        psi = np.arctan2(dy, dx)
        speed_sq = dx ** 2 + dy ** 2
        kappa = (dx * d2y - dy * d2x) / (speed_sq ** 1.5 + 1e-12)

        return xy, s_uniform, psi, kappa

    # ─────────────────────────────────────────────────────────────
    # Curvature enforcement
    # ─────────────────────────────────────────────────────────────

    def _smooth_curvature(self, xy, max_kappa):
        """Iteratively smooth until curvature constraint is met."""
        sm_xy = xy
        for sigma in np.linspace(2.0, 30.0, 20):
            sm_x = gaussian_filter1d(xy[:, 0], sigma=sigma, mode='wrap')
            sm_y = gaussian_filter1d(xy[:, 1], sigma=sigma, mode='wrap')
            sm_xy = np.column_stack([sm_x, sm_y])
            s, psi, kappa = self._finite_diff_properties(sm_xy)
            if np.all(np.abs(kappa) <= max_kappa * 1.05):
                return sm_xy, s, psi, kappa
        # Best effort with last sigma
        s, psi, kappa = self._finite_diff_properties(sm_xy)
        return sm_xy, s, psi, kappa

    def _finite_diff_properties(self, xy):
        """Compute s, psi, kappa from discrete (x,y) using finite differences."""
        dx = np.roll(xy[:, 0], -1) - xy[:, 0]
        dy = np.roll(xy[:, 1], -1) - xy[:, 1]
        ds_seg = np.sqrt(dx ** 2 + dy ** 2)

        s = np.concatenate([[0.0], np.cumsum(ds_seg[:-1])])
        psi = np.arctan2(dy, dx)

        dpsi = np.roll(psi, -1) - psi
        dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
        kappa = dpsi / (ds_seg + 1e-12)

        return s, psi, kappa

    # ─────────────────────────────────────────────────────────────
    # Track widths
    # ─────────────────────────────────────────────────────────────

    def _generate_widths(self, kappa, n):
        """Generate track half-widths (right, left)."""
        min_hw = self.min_track_width / 2.0
        max_hw = self.max_track_width / 2.0

        if not self.variable_width:
            hw = float(self.rng.uniform(min_hw, max_hw))
            hw_arr = np.full(n, hw)
            max_safe_hw = 0.9 / (np.abs(kappa) + 1e-6)
            hw_arr = np.minimum(hw_arr, max_safe_hw)
            hw_arr = np.clip(hw_arr, min_hw, max_hw)
            return hw_arr, hw_arr.copy()

        # Variable: wider on straights, narrower in turns
        abs_k = np.abs(kappa)
        k_ref = np.percentile(abs_k, 95) + 1e-12
        k_norm = np.clip(abs_k / k_ref, 0.0, 1.0)

        hw = max_hw - k_norm * (max_hw - min_hw)
        hw = gaussian_filter1d(hw, sigma=max(5, n // 50), mode='wrap')
        hw = np.clip(hw, min_hw, max_hw)

        # Prevent inner-boundary self-intersection at tight turns.
        # The offset boundary cusps when hw * |kappa| >= 1.
        max_safe_hw = 0.9 / (abs_k + 1e-6)
        hw = np.minimum(hw, max_safe_hw)
        hw = np.clip(hw, min_hw, max_hw)
        hw = gaussian_filter1d(hw, sigma=max(5, n // 50), mode='wrap')
        hw = np.clip(hw, min_hw, max_hw)

        return hw.copy(), hw.copy()

    # ─────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────

    def _compute_normals(self, psi):
        """Unit normals pointing left of the travel direction."""
        return np.column_stack([-np.sin(psi), np.cos(psi)])

    def _boundaries_valid(self, centerline, total_widths):
        """Reject tracks where non-adjacent track segments overlap."""
        n = len(centerline)
        step = max(1, n // 300)
        cl = centerline[::step]
        w = total_widths[::step]
        m = len(cl)

        min_gap = 5
        for i in range(m):
            for j in range(i + min_gap, m):
                idx_gap = min(j - i, m - (j - i))
                if idx_gap < min_gap:
                    continue
                dist = math.hypot(
                    cl[i, 0] - cl[j, 0], cl[i, 1] - cl[j, 1])
                needed = (w[i] + w[j]) / 2.0
                if dist < needed:
                    return False
        return True

    # ─────────────────────────────────────────────────────────────
    # Resampling
    # ─────────────────────────────────────────────────────────────

    def _downsample(self, xy, s, psi, kappa, target_ds):
        """Downsample track data to *target_ds* arc-length spacing."""
        total = s[-1]
        n_out = max(int(total / target_ds), 10)
        s_out = np.linspace(0, total, n_out, endpoint=False)

        x_out = np.interp(s_out, s, xy[:, 0])
        y_out = np.interp(s_out, s, xy[:, 1])
        psi_out = np.interp(s_out, s, np.unwrap(psi))
        kappa_out = np.interp(s_out, s, kappa)

        return np.column_stack([x_out, y_out]), s_out, psi_out, kappa_out

    # ─────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────

    def _render_occupancy(self, left_boundary, right_boundary):
        """Render the track as black wall lines on a white background.

        Matches the style of existing F1TENTH map PNGs (white = free space,
        black lines = walls along the track boundaries).

        Returns (PIL.Image, origin_list, effective_resolution).
        """
        all_pts = np.vstack([left_boundary, right_boundary])
        x_min, y_min = all_pts.min(axis=0) - self.margin_meters
        x_max, y_max = all_pts.max(axis=0) + self.margin_meters

        world_w = x_max - x_min
        world_h = y_max - y_min

        res = self.resolution
        img_w = int(math.ceil(world_w / res))
        img_h = int(math.ceil(world_h / res))

        # Auto-coarsen if image would be too large
        if max(img_w, img_h) > self.max_image_dim:
            res = max(world_w, world_h) / self.max_image_dim
            img_w = int(math.ceil(world_w / res))
            img_h = int(math.ceil(world_h / res))

        origin = [float(x_min), float(y_min), 0.0]

        # White background (free space), black wall lines
        img = Image.new("L", (img_w, img_h), 255)
        draw = ImageDraw.Draw(img)

        def to_px(pts):
            cols = (pts[:, 0] - origin[0]) / res
            rows = (img_h - 1) - (pts[:, 1] - origin[1]) / res
            return list(zip(cols.tolist(), rows.tolist()))

        left_px = to_px(left_boundary)
        right_px = to_px(right_boundary)

        # Wall thickness is doubled: the interior fill (step 2) erases the
        # inward half, leaving ~wall_width/2 visible outward-facing wall.
        wall_width = max(6, int(round(0.5 / res)))
        r = wall_width // 2  # radius for joint circles

        # Step 1: Draw thick black boundary walls with filled circles at
        # every vertex to eliminate gaps at segment joints.
        draw.line(left_px + [left_px[0]], fill=0, width=wall_width)
        for px, py in left_px:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=0)
        draw.line(right_px + [right_px[0]], fill=0, width=wall_width)
        for px, py in right_px:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

        # Step 2: Fill the track interior with white.  This erases any
        # triangular overlap artifacts at sharp turns where the inner
        # boundary self-intersects, while preserving the outward-facing
        # wall pixels.
        corridor = left_px + list(reversed(right_px))
        draw.polygon(corridor, fill=255)

        return img, origin, res

    # ─────────────────────────────────────────────────────────────
    # File I/O
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _write_image(path, img):
        img.save(path)

    @staticmethod
    def _write_yaml(path, image_name, resolution, origin):
        with open(path, "w") as f:
            f.write(f"image: {image_name}\n")
            f.write(f"resolution: {resolution}\n")
            f.write(f"origin: [{origin[0]},{origin[1]}, {origin[2]}]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.45\n")
            f.write("free_thresh: 0.196\n")

    def _write_raceline_csv(self, path, s, xy, psi, kappa):
        with open(path, "w") as f:
            f.write(f"# {uuid.uuid4()}\n")
            f.write("# generated_by_track_generator\n")
            f.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
            for i in range(len(s)):
                f.write(
                    f"{s[i]:.7f};{xy[i, 0]:.7f};{xy[i, 1]:.7f};"
                    f"{psi[i]:.7f};{kappa[i]:.7f};"
                    f"{self.raceline_speed:.7f};0.0000000\n"
                )

    @staticmethod
    def _write_centerline_csv(path, xy, w_right, w_left):
        with open(path, "w") as f:
            f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
            for i in range(len(xy)):
                f.write(
                    f"{xy[i, 0]}, {xy[i, 1]}, {w_right[i]}, {w_left[i]}\n")


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = TrackGenerator(
        min_track_length=100,
        max_track_length=600,
    )
    name = gen.generate("test_track")
    print(f"Generated track: {name}")
    print(f"Files in maps/{name}/:")
    for fn in sorted(os.listdir(os.path.join("maps", name))):
        print(f"  {fn}")
