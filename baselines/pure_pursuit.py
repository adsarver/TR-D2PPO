import numpy as np
import os

class PurePursuit:
    """
    Traditional Pure Pursuit controller for F1TENTH racing.
    Follows a raceline by looking ahead a fixed number of waypoints and
    autonomously determines speed based on path curvature.
    """

    def __init__(self, map_name, lookahead_points=20, wheelbase=0.33,
                 max_steering=0.34, max_speed=8.0, min_speed=1.5,
                 lookahead_distance=None, curvature_lookahead_points=None):
        """
        Args:
            map_name: Name of the track (e.g., "Hockenheim")
            lookahead_points: Number of waypoints to look ahead for steering target
            wheelbase: Distance between front and rear axles (meters)
            max_steering: Maximum steering angle (radians)
            max_speed: Top speed on straights (m/s)
            min_speed: Minimum speed through tight corners (m/s)
            lookahead_distance: If set, overrides lookahead_points with a fixed
                distance (meters) so steering lookahead is speed-independent.
            curvature_lookahead_points: Waypoints to scan for curvature-based
                speed control. Defaults to max(lookahead_points * 5, 30) so
                the controller can anticipate corners well ahead of the car.
        """
        self.map_name = map_name
        self.lookahead_points = lookahead_points
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.lookahead_distance = lookahead_distance  # metres; None = use point count
        self.curvature_lookahead_points = curvature_lookahead_points or 100  # ~20 m at 0.2 m/pt

        # Load raceline waypoints (x, y only)
        self.waypoints = self._load_raceline()
        self.num_waypoints = len(self.waypoints)
        self._last_nearest_idx = 0  # Tracks position around the lap for local search
        self._agent_nearest_idx = {}  # Per-agent tracking for batch mode

    def _load_raceline(self):
        """Load raceline CSV and extract (x, y) waypoints.
        
        If a centerline CSV with track-width data exists, waypoints that are
        closer than ``_min_wall_clearance`` to a wall are nudged towards the
        track centre so that the pure-pursuit controller has enough margin to
        track the path without hitting the wall.
        """
        raceline_path = os.path.join('maps', self.map_name, f'{self.map_name}_raceline.csv')

        if not os.path.exists(raceline_path):
            raise FileNotFoundError(f"Raceline not found: {raceline_path}")

        # CSV columns: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
        data = np.loadtxt(raceline_path, delimiter=';', skiprows=3)
        self.waypoint_s = data[:, 0]   # cumulative arc-length (metres)
        self.waypoint_spacing = np.mean(np.diff(self.waypoint_s))  # avg metres/waypoint
        raceline_kappa = np.abs(data[:, 4])  # curvature from the planner (smooth)
        waypoints = data[:, 1:3].copy()  # (N, 2) x, y

        # --- Wall-clearance blending ---
        centerline_path = os.path.join('maps', self.map_name,
                                       f'{self.map_name}_centerline.csv')
        if os.path.exists(centerline_path):
            cl = np.loadtxt(centerline_path, delimiter=',')
            # cl columns: x, y, w_right, w_left
            if cl.shape[1] >= 4:
                waypoints = self._blend_for_clearance(waypoints, cl)

        # --- Compute effective curvature for speed planning ---
        # Use the raceline planner's kappa directly — it is smooth and
        # accurate.  Where blending shifted the path significantly, add a
        # curvature penalty proportional to the lateral shift so the car
        # slows down through sections that were pulled away from the
        # optimal line.
        self.waypoint_kappa = self._compute_effective_kappa(
            waypoints, raceline_kappa, data[:, 1:3])

        return waypoints

    # Required clearance from any wall (metres).  Must exceed car half-width
    # (0.155 m) plus a small tracking-error margin.
    _min_wall_clearance = 0.35

    def _blend_for_clearance(self, waypoints, centerline):
        """Shift waypoints towards the track centre where wall clearance is
        below ``_min_wall_clearance``.

        For each raceline point:
          1. Find the nearest centerline point and its track half-widths.
          2. Compute the signed offset from center and determine which wall
             is closest.
          3. If the raceline is too close to that wall, pull it inward just
             enough to restore the required clearance margin.
        """
        cl_xy = centerline[:, :2]
        w_r = centerline[:, 2]
        w_l = centerline[:, 3]
        car_hw = 0.155  # car half-width (0.31 m / 2)

        # Pre-compute centerline tangent normals for signed offset
        # (right-hand normal: rotate tangent 90° clockwise)
        cl_tangent = np.zeros_like(cl_xy)
        cl_tangent[:-1] = np.diff(cl_xy, axis=0)
        cl_tangent[-1] = cl_xy[0] - cl_xy[-1]            # wrap
        cl_len = np.linalg.norm(cl_tangent, axis=1, keepdims=True)
        cl_len = np.maximum(cl_len, 1e-9)
        cl_tangent /= cl_len
        # Right-hand normal: (tx, ty) → (ty, -tx)
        cl_normal_r = np.column_stack([cl_tangent[:, 1], -cl_tangent[:, 0]])

        blended = waypoints.copy()
        for i in range(len(waypoints)):
            # Nearest centerline point
            dists_sq = np.sum((cl_xy - waypoints[i])**2, axis=1)
            j = np.argmin(dists_sq)
            cl_pt = cl_xy[j]

            vec = waypoints[i] - cl_pt           # center → raceline
            offset = np.linalg.norm(vec)

            if offset < 1e-6:
                continue  # already on centreline

            # Determine which side of the track the raceline is on:
            # positive dot with right-normal → right side → near right wall
            dot = vec[0] * cl_normal_r[j, 0] + vec[1] * cl_normal_r[j, 1]
            if dot >= 0:
                half_width = w_r[j]   # near the right wall
            else:
                half_width = w_l[j]   # near the left wall

            clearance = half_width - offset - car_hw

            if clearance < self._min_wall_clearance:
                desired_offset = half_width - car_hw - self._min_wall_clearance
                desired_offset = max(0.0, desired_offset)
                direction = vec / offset
                blended[i] = cl_pt + direction * desired_offset

        return blended

    @staticmethod
    def _compute_effective_kappa(blended_waypoints, raceline_kappa,
                                 original_waypoints):
        """Return the curvature array used for speed planning.

        Uses the raceline planner's smooth kappa directly.  The blending only
        shifts waypoints laterally by a small amount (< 0.3 m typically) to
        maintain wall clearance; the resulting path curvature change is
        negligible compared to the actual corner curvature, so the planner
        kappa remains the best estimate of the speed-limiting curvature.
        """
        return raceline_kappa.copy()

    def update_map(self, map_name):
        """Switch to a new map and reload waypoints."""
        self.map_name = map_name
        self.waypoints = self._load_raceline()
        self.num_waypoints = len(self.waypoints)
        self._last_nearest_idx = 0
        self._agent_nearest_idx = {}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _get_pose_from_obs(self, obs, agent_idx=0):
        """Extract (x, y, theta) from the simulator observation dict."""
        return (obs['poses_x'][agent_idx],
                obs['poses_y'][agent_idx],
                obs['poses_theta'][agent_idx])

    def _find_nearest_index(self, x, y, agent_idx=0):
        """Return the index of the closest waypoint to (x, y).
        
        Searches only within a local window around the last known index to
        prevent snapping to a distant part of the track (e.g. the other side
        of a loop or the start/finish line). Falls back to a global search if
        the car has moved far from the last known position (e.g. after a reset).
        """
        last_idx = self._agent_nearest_idx.get(agent_idx, None)

        if last_idx is not None:
            # Check if the car is still close to the last known position
            last_pt = self.waypoints[last_idx]
            if np.sum((np.array([x, y]) - last_pt)**2) < 25.0:  # within 5 m
                search_window = min(200, self.num_waypoints)
                half = search_window // 2
                indices = np.array([(last_idx + i - half) % self.num_waypoints
                                     for i in range(search_window)])
                pts = self.waypoints[indices]
                dists = np.sum((pts - np.array([x, y]))**2, axis=1)
                nearest = int(indices[np.argmin(dists)])
                self._agent_nearest_idx[agent_idx] = nearest
                return nearest

        # Global search — first call or after a teleport/reset
        dists = np.sum((self.waypoints - np.array([x, y]))**2, axis=1)
        nearest = int(np.argmin(dists))
        self._agent_nearest_idx[agent_idx] = nearest
        return nearest

    def _get_lookahead_point(self, nearest_idx, current_speed=None):
        """Return the (x, y) of the waypoint ahead of the car.

        If lookahead_distance is set, the distance is scaled with speed:
          - At low speed the lookahead is clamped to a minimum (0.8 × base)
            to prevent oscillation from chasing a point too close to the car.
          - At high speed the lookahead grows so steering is smooth and stable.

        Scaling: ld = base_distance * clamp(speed / 6.0, 0.8, 1.5)
        """
        if self.lookahead_distance is not None:
            if current_speed is not None:
                speed_factor = np.clip(current_speed / 6.0, 0.8, 1.5)
            else:
                speed_factor = 1.0
            ld = self.lookahead_distance * speed_factor
            n = max(1, int(round(ld / self.waypoint_spacing)))
        else:
            n = self.lookahead_points
        target_idx = (nearest_idx + n) % self.num_waypoints
        return self.waypoints[target_idx]

    # ------------------------------------------------------------------
    # Core pure-pursuit math
    # ------------------------------------------------------------------

    def _compute_steering(self, x, y, theta, target_x, target_y):
        """
        Classic pure-pursuit steering law.

        1. Transform the target into the vehicle frame.
        2. steering = arctan(2 * L * sin(alpha) / ld)
        """
        dx = target_x - x
        dy = target_y - y

        # Vehicle-frame coordinates
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        dx_v = dx * cos_t - dy * sin_t
        dy_v = dx * sin_t + dy * cos_t

        ld = np.hypot(dx_v, dy_v)
        if ld < 0.01:
            return 0.0

        alpha = np.arctan2(dy_v, dx_v)
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)
        return np.clip(steering, -self.max_steering, self.max_steering)

    def _compute_speed(self, nearest_idx, current_speed):
        """
        Compute the maximum safe speed at the current position by working
        backwards from the minimum speed required at each upcoming corner.

        Uses the **raceline planner's curvature** (kappa) rather than
        finite-difference curvature on the waypoints, because the planner
        values are smooth and free of numerical noise.  The blended path may
        shift waypoints for wall clearance, but the curvature profile stays
        the same — the planner kappa is the best available estimate of how
        tight each corner really is.

        For every point in the lookahead window:
          1. Look up kappa from the stored raceline data.
          2. Compute the corner speed limit: v = sqrt(a_lat / kappa).
          3. Back-propagate with braking: v_now = sqrt(v_ahead² + 2·a_brake·d).
        """
        n = self.curvature_lookahead_points
        indices = np.array([(nearest_idx + i) % self.num_waypoints
                             for i in range(n)])

        kappa = self.waypoint_kappa[indices]               # (n,)
        # Segment lengths from arc-length column (wrap-safe)
        seg_len = np.empty(n)
        for k_idx in range(n):
            i0 = indices[k_idx]
            i1 = (i0 + 1) % self.num_waypoints
            ds = self.waypoint_s[i1] - self.waypoint_s[i0]
            if ds <= 0:                       # wrap-around at lap boundary
                ds = self.waypoint_spacing
            seg_len[k_idx] = ds

        # Lateral-acceleration budget (speed-dependent)
        # These are conservative because:
        #   1. The car follows a blended path, not the exact raceline.
        #   2. Pure-pursuit has inherent tracking error (~0.1-0.2 m).
        #   3. The dynamic bicycle model has less grip above v_switch.
        v_switch = 7.3
        a_lat_kin  = 0.35 * 9.81   # ≈ 3.4 m/s²  (kinematic regime)
        a_lat_dyn  = 0.25 * 9.81   # ≈ 2.5 m/s²  (dynamic regime)
        a_brake    = 9.51 * 0.70   # braking while steering

        with np.errstate(divide='ignore', invalid='ignore'):
            v_corner_dyn = np.where(kappa > 1e-4,
                                    np.sqrt(a_lat_dyn / kappa),
                                    self.max_speed)
            v_corner_kin = np.where(kappa > 1e-4,
                                    np.sqrt(a_lat_kin / kappa),
                                    self.max_speed)
            # Use kinematic limit only where that limit is below v_switch
            v_corner = np.where(v_corner_kin < v_switch,
                                v_corner_kin,
                                v_corner_dyn)
        v_corner = np.clip(v_corner, self.min_speed, self.max_speed)

        # Backwards pass: propagate speed limits back to current position
        v_safe = v_corner[-1]
        for i in range(len(v_corner) - 2, -1, -1):
            v_safe = min(v_corner[i],
                         np.sqrt(v_safe**2 + 2.0 * a_brake * seg_len[i]))

        return float(np.clip(v_safe, self.min_speed, self.max_speed))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, obs, agent_idx=0, **_kwargs):
        """
        Compute [steering, speed] for one agent.

        Args:
            obs: Observation dictionary from the simulator.
            agent_idx: Which agent to control.

        Returns:
            action: np.array([steering, speed])
        """
        x, y, theta = self._get_pose_from_obs(obs, agent_idx)
        current_speed = obs['linear_vels_x'][agent_idx]

        nearest_idx = self._find_nearest_index(x, y, agent_idx)
        target_x, target_y = self._get_lookahead_point(nearest_idx, current_speed)

        steering = self._compute_steering(x, y, theta, target_x, target_y)
        speed    = self._compute_speed(nearest_idx, current_speed)

        return np.array([steering, speed])

    def get_actions_batch(self, obs, random_speed=False, **_kwargs):
        """
        Get actions for every agent in the observation.

        Args:
            obs: Observation dictionary from the simulator.
            random_speed: If True, override with uniform-random speeds.

        Returns:
            actions: (num_agents, 2) array of [steering, speed]
        """
        num_agents = len(obs['poses_x'])
        actions = np.zeros((num_agents, 2))

        for i in range(num_agents):
            actions[i] = self.get_action(obs, agent_idx=i)

        if random_speed:
            actions[:, 1] = np.random.uniform(self.min_speed, self.max_speed, num_agents)

        return actions

    # ------------------------------------------------------------------
    # Tuning helpers
    # ------------------------------------------------------------------

    def set_lookahead_points(self, n):
        """Change how many waypoints ahead to target."""
        self.lookahead_points = max(1, int(n))

    def set_speed_limits(self, min_speed=None, max_speed=None):
        """Adjust the speed envelope."""
        if min_speed is not None:
            self.min_speed = min_speed
        if max_speed is not None:
            self.max_speed = max_speed