"""
Fan-of-trajectories MPC planner adapted from jour.py (ROS 2 node) to work
directly with the f110_gym observation dictionary.

Exposes the same interface as PurePursuit / GapFollow:
    __init__(map_name, ...)
    get_action(obs, agent_idx=0)
    get_actions_batch(obs)
    update_map(map_name)
"""

import math
import os
import numpy as np


class MPCAgent:
    """
    Raceline-following + LiDAR-reactive fan-of-trajectories planner.

    At each step the planner:
      1. Finds the nearest raceline waypoint and builds a short reference
         trajectory (horizon steps ahead).
      2. Analyses the LiDAR scan for gap-follow steering and checks for
         obstacles ahead.
      3. Generates a fan of candidate constant-steering trajectories around
         the raceline (or the gap heading when an obstacle is close).
      4. Forward-simulates each candidate, scores it against the reference
         and LiDAR clearance, and picks the best.
      5. Applies emergency-recovery / overtake bias when the front cone is
         very close.
    """

    # ------------------------------------------------------------------ #
    # Construction / map loading
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        map_name,
        wheelbase=0.33,
        max_steering=0.34,
        max_speed=14.0,
        min_speed=0.0,
        max_accel=3.0,
        speed_scale=1.2,
        # MPC horizon
        horizon=8,
        dt=0.1,
        step_length=0.2,
        # Gap-follow
        downsample_gap=5,
        max_sight=10.0,
        gap_threshold=15.0,
        # Emergency / overtake
        emergency_dist=1.2,
        recovery_speed=0.8,
        recovery_steer=0.30,
        front_cone_deg=10.0,
        side_cone_min_deg=60.0,
        side_cone_max_deg=110.0,
        overtake_front_thresh=3.5,
        overtake_bias_rad=0.20,
        # LiDAR geometry (must match env)
        num_beams=1080,
        fov=4.7,
        **_kwargs,
    ):
        self.map_name = map_name
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.min_steer = -max_steering
        self.max_steer = max_steering
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_accel = max_accel
        self.speed_scale = speed_scale

        # MPC
        self.horizon = horizon
        self.dt = dt
        self.step_length = step_length
        self.state_size = 4  # x, y, v, yaw

        # Gap-follow
        self.downsample_gap = downsample_gap
        self.max_sight = max_sight
        self.gap_threshold = gap_threshold

        # Emergency / overtake
        self.emergency_dist = emergency_dist
        self.recovery_speed = recovery_speed
        self.recovery_steer = recovery_steer
        self.front_cone_deg = front_cone_deg
        self.side_cone_min_deg = side_cone_min_deg
        self.side_cone_max_deg = side_cone_max_deg
        self.overtake_front_thresh = overtake_front_thresh
        self.overtake_bias_rad = overtake_bias_rad

        # LiDAR geometry
        self.num_beams = num_beams
        self.fov = fov
        self._angle_increment = fov / num_beams
        self._angle_min = -fov / 2.0

        # Internal state (reset per-step)
        self.obstacle_ahead = False
        self.raceline_steer = 0.0
        self.front_clearance = max_sight

        # Load map
        self._load_map(map_name)

    # ------------------------------------------------------------------ #
    # Map / raceline loading
    # ------------------------------------------------------------------ #

    def _load_map(self, map_name):
        """Load raceline waypoints for the given map."""
        raceline_path = os.path.join("maps", map_name, f"{map_name}_raceline.csv")
        if not os.path.exists(raceline_path):
            raise FileNotFoundError(f"Raceline not found: {raceline_path}")

        data = np.loadtxt(raceline_path, delimiter=";", skiprows=3)
        # columns: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
        n = data.shape[0]
        self.waypoints = np.zeros((n, 6), dtype=float)
        self.waypoints[:, 0] = data[:, 0]  # s (arc-length)
        self.waypoints[:, 1] = data[:, 1]  # x
        self.waypoints[:, 2] = data[:, 2]  # y
        self.waypoints[:, 3] = data[:, 3]  # yaw
        self.waypoints[:, 5] = data[:, 5]  # speed

        self.waypoint_kappa = np.abs(data[:, 4])  # |kappa| rad/m
        self.speed_profile = self.waypoints[:, 5] * self.speed_scale
        self.num_waypoints = n

        # Pre-compute segment lengths from arc-length data
        self._seg_len = np.empty(n)
        spacing = np.mean(np.diff(data[:, 0]))
        for i in range(n):
            ds = data[(i + 1) % n, 0] - data[i, 0]
            self._seg_len[i] = ds if ds > 0 else spacing

    def update_map(self, map_name):
        """Switch to a new map and reload waypoints."""
        self.map_name = map_name
        self._load_map(map_name)

    # ------------------------------------------------------------------ #
    # Curvature-based speed planning  (like Pure Pursuit, proven reliable)
    # ------------------------------------------------------------------ #

    def _compute_safe_speed(self, nearest_idx):
        """Backwards-propagating braking model using raceline curvature.

        Uses the same dual-regime lateral-acceleration model as PurePursuit:
          - Kinematic regime (low speed): a_lat_kin = 0.35 g ≈ 3.4 m/s²
          - Dynamic regime (high speed):  a_lat_dyn = 0.25 g ≈ 2.5 m/s²
          - Switch at v_switch = 7.3 m/s

        For each waypoint in the lookahead window:
          1. Compute the corner speed limit from curvature: v = sqrt(a_lat / kappa).
          2. Back-propagate with braking: v = sqrt(v_ahead² + 2·a·d).
        Returns the maximum safe speed at the current position.
        """
        lookahead_pts = 100  # ~20 m ahead
        n = self.num_waypoints
        indices = np.array([(nearest_idx + i) % n for i in range(lookahead_pts)])

        kappa = self.waypoint_kappa[indices]
        seg_len = self._seg_len[indices]

        # Dual-regime lateral-acceleration budget (matches PurePursuit)
        v_switch = 7.3
        a_lat_kin = 0.35 * 9.81   # ≈ 3.4 m/s² (kinematic, tighter corners)
        a_lat_dyn = 0.25 * 9.81   # ≈ 2.5 m/s² (dynamic, high-speed)
        a_brake = 9.51 * 0.70     # braking while cornering (matches PP)

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

        # Backwards pass
        v_safe = float(v_corner[-1])
        for i in range(len(v_corner) - 2, -1, -1):
            v_safe = min(float(v_corner[i]),
                         math.sqrt(v_safe ** 2 + 2.0 * a_brake * seg_len[i]))

        return float(np.clip(v_safe, self.min_speed, self.max_speed))

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_nearest_point(self, pt, traj):
        """Nearest-segment projection.  Returns (proj, dist, t, seg_idx)."""
        diffs = traj[1:, :] - traj[:-1, :]
        l2s = (diffs ** 2).sum(axis=1)
        dots = ((pt - traj[:-1]) * diffs).sum(axis=1)
        t_vals = np.clip(dots / np.maximum(l2s, 1e-12), 0.0, 1.0)
        projections = traj[:-1] + (t_vals[:, None] * diffs)
        dists = np.linalg.norm(pt - projections, axis=1)
        min_idx = int(np.argmin(dists))
        return projections[min_idx], dists[min_idx], t_vals[min_idx], min_idx

    # ------------------------------------------------------------------ #
    # LiDAR helpers  (replacing ROS LaserScan message queries)
    # ------------------------------------------------------------------ #

    def _angle_to_index(self, angle_rad, n_beams):
        idx = int(round((angle_rad - self._angle_min) / self._angle_increment))
        return max(0, min(n_beams - 1, idx))

    def _get_scan_sector(self, ranges, a_min, a_max):
        """Return (sector_ranges, i0, i1) for the angular window [a_min, a_max]."""
        n = len(ranges)
        if a_max < a_min:
            a_min, a_max = a_max, a_min
        i0 = self._angle_to_index(a_min, n)
        i1 = self._angle_to_index(a_max, n)
        if i1 < i0:
            i0, i1 = i1, i0
        i1 = min(i1 + 1, n)
        return ranges[i0:i1].copy(), i0, i1

    def _cone_min(self, ranges, deg_min, deg_max):
        a0 = math.radians(deg_min)
        a1 = math.radians(deg_max)
        if a1 < a0:
            a0, a1 = a1, a0
        n = len(ranges)
        angles = self._angle_min + np.arange(n, dtype=np.float32) * self._angle_increment
        mask = (angles >= a0) & (angles <= a1)
        vals = ranges[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.min(vals)) if vals.size else float("inf")

    def _front_min(self, ranges, half_deg):
        return self._cone_min(ranges, -half_deg, +half_deg)

    # ------------------------------------------------------------------ #
    # Gap-follow
    # ------------------------------------------------------------------ #

    def _preprocess_scan(self, sector_ranges):
        num_bins = max(1, int(len(sector_ranges) / self.downsample_gap))
        proc = np.zeros(num_bins)
        for i in range(num_bins):
            window = sector_ranges[i * self.downsample_gap:(i + 1) * self.downsample_gap]
            proc[i] = float(np.mean(window)) if len(window) else self.max_sight
        return np.clip(proc, 0.0, self.max_sight)

    def _get_max_gap(self, free_ranges, current_speed):
        longest = 0
        current = 0
        end_idx = 0
        safe_dist = 0.7 + 0.4 * current_speed
        start_idx = 0
        for i in range(len(free_ranges)):
            if free_ranges[i] > safe_dist:
                current += 1
                if current > longest:
                    longest = current
                    end_idx = i + 1
                    start_idx = end_idx - longest
            else:
                current = 0
        return start_idx, end_idx

    def _compute_gap_steer_and_speed(self, ranges, gap_start, gap_end,
                                      gap_sector_start_idx, current_x,
                                      current_y, current_yaw, current_speed):
        """Compute gap-follow steering and speed from raw scan + gap indices."""
        fwd_sector, _, _ = self._get_scan_sector(
            ranges, -math.radians(10.0), math.radians(10.0))
        fwd_sector = fwd_sector[np.isfinite(fwd_sector)]
        min_forward = float(np.min(fwd_sector)) if fwd_sector.size else self.max_sight
        self.front_clearance = min_forward

        cx = self.waypoints[:, 1]
        cy = self.waypoints[:, 2]
        pt = np.array([current_x, current_y])
        proj, cross_dist, _, idx = self._get_nearest_point(pt, np.vstack((cx, cy)).T)

        # Look a few points ahead so the steering target is in front, not beside the car
        lookahead_idx = (idx + max(3, int(abs(current_speed) * 0.15 / max(self.step_length, 0.01)))) % self.num_waypoints
        target_x = float(cx[lookahead_idx])
        target_y = float(cy[lookahead_idx])

        # Pure-pursuit-style steering toward the raceline lookahead point
        dx = target_x - current_x
        dy = target_y - current_y
        cos_t = math.cos(-current_yaw)
        sin_t = math.sin(-current_yaw)
        dx_v = dx * cos_t - dy * sin_t
        dy_v = dx * sin_t + dy * cos_t
        ld = math.hypot(dx_v, dy_v)
        if ld > 0.01:
            alpha = math.atan2(dy_v, dx_v)
            raceline_steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        else:
            raceline_yaw = float(self.waypoints[idx, 3])
            raceline_steer = self._normalize_angle(raceline_yaw - current_yaw)
        raceline_steer = float(np.clip(raceline_steer, self.min_steer, self.max_steer))
        self.raceline_steer = raceline_steer

        free_thresh = 3.0          # Only react to genuinely close obstacles
        max_dev = math.radians(40.0)
        self.obstacle_ahead = False
        steering_angle = raceline_steer

        if min_forward < free_thresh:
            self.obstacle_ahead = True
            best_i = 0.5 * (gap_start + gap_end)
            center_ray = int(best_i * self.downsample_gap) + gap_sector_start_idx
            center_ray = max(0, min(len(ranges) - 1, center_ray))
            gap_steer = float(self._angle_min + center_ray * self._angle_increment)

            angle_diff = abs(self._normalize_angle(gap_steer - raceline_steer))
            if angle_diff <= max_dev:
                steering_angle = gap_steer
            else:
                self.obstacle_ahead = False
                steering_angle = raceline_steer

        # Gap-follow speed ladder — only active for very close obstacles
        if min_forward < 1.0:
            velocity = max(2.0, 0.3 * self.max_speed)
        elif min_forward < 2.0:
            velocity = max(4.0, 0.5 * self.max_speed)
        elif min_forward < free_thresh:
            frac = (min_forward - 2.0) / max(free_thresh - 2.0, 0.01)
            velocity = (0.5 + 0.5 * frac) * self.max_speed
        else:
            velocity = self.max_speed

        velocity = min(float(velocity), float(self.max_speed))
        return float(steering_angle), float(velocity)

    # ------------------------------------------------------------------ #
    # Reference trajectory
    # ------------------------------------------------------------------ #

    def _compute_reference_traj(self, x, y, yaw, v):
        """Build a spatial reference from the raceline.

        The reference carries the raceline (x, y, yaw).  The speed row
        stores the curvature-safe speed at each lookahead point so the
        cost function can penalise deviations from the target speed.
        """
        cx = self.waypoints[:, 1]
        cy = self.waypoints[:, 2]
        cyaw = self.waypoints[:, 3].copy()
        n_course = len(cx)

        _, _, _, idx = self._get_nearest_point(
            np.array([x, y]), np.vstack((cx, cy)).T)
        self._nearest_idx = idx   # cached for speed planning

        ref_traj = np.zeros((self.state_size, self.horizon + 1))

        travel_dist = abs(v) * self.dt
        d_index = max(travel_dist / self.step_length, 1.0)

        idx_list = int(idx) + np.insert(
            np.cumsum(np.repeat(d_index, self.horizon)), 0, 0
        ).astype(int)
        idx_list = idx_list % n_course

        ref_traj[0, :] = cx[idx_list]
        ref_traj[1, :] = cy[idx_list]
        # The speed row uses the curvature-safe speed; this will be
        # passed to _generate_candidate_controls as v_target.
        safe_speed = self._compute_safe_speed(idx)
        ref_traj[2, :] = safe_speed

        angle_thresh = 4.5
        cyaw2 = cyaw.copy()
        for i in range(len(cyaw2)):
            if cyaw2[i] - yaw > angle_thresh:
                cyaw2[i] -= 2 * np.pi
            if yaw - cyaw2[i] > angle_thresh:
                cyaw2[i] += 2 * np.pi
        ref_traj[3, :] = cyaw2[idx_list]
        return ref_traj

    # ------------------------------------------------------------------ #
    # Candidate generation + simulation
    # ------------------------------------------------------------------ #

    def _propagate_local(self, x, y, yaw, v, a, delta):
        dt = self.dt
        v_next = np.clip(v + a * dt, self.min_speed, self.max_speed)
        x_next = x + v_next * math.cos(yaw) * dt
        y_next = y + v_next * math.sin(yaw) * dt
        yaw_next = yaw + v_next / self.wheelbase * math.tan(delta) * dt
        return x_next, y_next, yaw_next, v_next

    def _generate_candidate_controls(self, v_target, current_speed):
        """Generate a fan of steering candidates with P-control accel
        toward the curvature-safe speed target.
        """
        N = self.horizon
        if not self.obstacle_ahead:
            num_steer = 15
            base = self.raceline_steer
            fan_half = np.deg2rad(15.0)
        else:
            num_steer = 21
            base = getattr(self, '_gap_steer', 0.0)
            fan_half = np.deg2rad(30.0)

        base = float(np.clip(base, self.min_steer, self.max_steer))
        left = np.clip(base + fan_half, self.min_steer, self.max_steer)
        right = np.clip(base - fan_half, self.min_steer, self.max_steer)
        if left < right:
            left, right = right, left

        steer_vals = np.linspace(right, left, num_steer)

        steering_candidates = []
        accel_candidates = []
        k_v = 3.0  # proportional gain

        a = k_v * (v_target - current_speed)
        a = float(np.clip(a, -self.max_accel, self.max_accel))

        for sv in steer_vals:
            steering_candidates.append(float(sv))
            accel_candidates.append(np.full(N, a, dtype=float))

        return np.array(steering_candidates), accel_candidates

    def _simulate_candidate(self, steer_const, accel_seq, current_speed):
        N = self.horizon
        x = np.zeros(N + 1)
        y = np.zeros(N + 1)
        yaw = np.zeros(N + 1)
        v = np.zeros(N + 1)
        v[0] = max(current_speed, 1.5)
        for t in range(N):
            x[t+1], y[t+1], yaw[t+1], v[t+1] = self._propagate_local(
                x[t], y[t], yaw[t], v[t], accel_seq[t], steer_const)
        return x, y, yaw, v

    def _check_collision_and_cost(self, x_local, y_local, v_local,
                                   steer_const,
                                   ref_traj, ranges, current_x, current_y,
                                   current_yaw):
        cos_y = math.cos(current_yaw)
        sin_y = math.sin(current_yaw)
        x_global = current_x + cos_y * x_local - sin_y * y_local
        y_global = current_y + sin_y * x_local + cos_y * y_local

        safety_radius = 0.55
        lidar_inflation = 0.30
        min_clearance = self.max_sight
        collision = False
        substeps = 5

        for t in range(len(x_local) - 1):
            x0, y0 = x_local[t], y_local[t]
            x1, y1 = x_local[t + 1], y_local[t + 1]
            for s in np.linspace(0.0, 1.0, substeps, endpoint=False):
                px = x0 + s * (x1 - x0)
                py = y0 + s * (y1 - y0)
                r = math.hypot(px, py)
                if r < 0.10:
                    continue
                theta = math.atan2(py, px)
                meas = self.max_sight
                if ranges is not None:
                    idx = int((theta - self._angle_min) / self._angle_increment)
                    if 0 <= idx < len(ranges):
                        rv = ranges[idx]
                        meas = rv if math.isfinite(rv) and rv > 0.0 else self.max_sight
                meas = max(meas - lidar_inflation, 0.0)
                clearance = meas - r
                min_clearance = min(min_clearance, clearance)
                if clearance < safety_radius:
                    collision = True
                    break
            if collision:
                break

        T = min(len(x_local), ref_traj.shape[1])
        track_err = 0.0
        for t in range(T):
            dx = x_global[t] - ref_traj[0, t]
            dy = y_global[t] - ref_traj[1, t]
            track_err += dx * dx + dy * dy

        forward_progress = x_local[-1]

        # Reward higher speed: penalise the gap between achieved speed and
        # max_speed so the MPC actually *wants* to accelerate.
        speed_reward = float(np.mean(v_local[1:]))  # avg speed over horizon

        if self.obstacle_ahead:
            w_track, w_steer, w_prog, w_clear, w_speed = 5.0, 0.2, 1.0, 8.0, 0.5
        else:
            w_track, w_steer, w_prog, w_clear, w_speed = 10.0, 1.0, 1.0, 0.0, 2.0

        cost = (w_track * track_err
                + w_steer * (steer_const ** 2)
                - w_prog * forward_progress
                - w_clear * max(min_clearance, 0.0)
                - w_speed * speed_reward)
        if collision:
            cost += 1e6
        return collision, cost

    def _run_candidate_mpc(self, current_x, current_y, current_yaw,
                            current_speed, ranges):
        ref_traj = self._compute_reference_traj(
            current_x, current_y, current_yaw, current_speed)

        # v_target comes from the curvature-safe speed planner
        v_target = float(ref_traj[2, 0])
        steering_cands, accel_cands = self._generate_candidate_controls(
            v_target, current_speed)

        best_cost = float("inf")
        best_idx = 0
        for idx, (sc, ac) in enumerate(zip(steering_cands, accel_cands)):
            x_l, y_l, _, v_l = self._simulate_candidate(sc, ac, current_speed)
            _, cost = self._check_collision_and_cost(
                x_l, y_l, v_l, sc, ref_traj, ranges,
                current_x, current_y, current_yaw)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        best_steer = float(steering_cands[best_idx])
        # Speed is the curvature-safe target; the MPC decides steering.
        best_speed = float(np.clip(v_target, self.min_speed, self.max_speed))
        return best_steer, best_speed

    # ------------------------------------------------------------------ #
    # Public API  (matches PurePursuit / GapFollow)
    # ------------------------------------------------------------------ #

    def get_action(self, obs, agent_idx=0, **_kwargs):
        """
        Compute [steering, speed] for one agent.

        Args:
            obs:        Observation dict from f110_gym.
            agent_idx:  Which agent in a multi-agent env.

        Returns:
            np.array([steering, speed])
        """
        current_x = obs['poses_x'][agent_idx]
        current_y = obs['poses_y'][agent_idx]
        current_yaw = obs['poses_theta'][agent_idx]
        current_speed = obs['linear_vels_x'][agent_idx]

        raw_scan = np.asarray(obs['scans'][agent_idx], dtype=np.float64)

        # --- Gap analysis ---
        sector_ranges, gap_i0, gap_i1 = self._get_scan_sector(
            raw_scan, -math.pi / 2.0, math.pi / 2.0)
        processed = self._preprocess_scan(sector_ranges)
        gap_start, gap_end = self._get_max_gap(processed, current_speed)

        gap_steer, gap_speed = self._compute_gap_steer_and_speed(
            raw_scan, gap_start, gap_end, gap_i0,
            current_x, current_y, current_yaw, current_speed)
        self._gap_steer = gap_steer

        # --- Overtake bias ---
        min_front = self._front_min(raw_scan, self.front_cone_deg)
        left_clear = self._cone_min(raw_scan, self.side_cone_min_deg, self.side_cone_max_deg)
        right_clear = self._cone_min(raw_scan, -self.side_cone_max_deg, -self.side_cone_min_deg)

        obstacle_ahead = min_front < self.overtake_front_thresh
        if obstacle_ahead and np.isfinite(left_clear) and np.isfinite(right_clear):
            bias = self.overtake_bias_rad if left_clear > right_clear else -self.overtake_bias_rad
            self._gap_steer = float(np.clip(
                gap_steer + bias, self.min_steer, self.max_steer))

        # --- Candidate MPC ---
        steer_cmd, speed_cmd = self._run_candidate_mpc(
            current_x, current_y, current_yaw, current_speed, raw_scan)

        # Blend toward gap steering when an actual obstacle is close
        if min_front < self.overtake_front_thresh:
            alpha = 0.35
            steer_cmd = float((1.0 - alpha) * steer_cmd + alpha * self._gap_steer)
            steer_cmd = float(np.clip(steer_cmd, self.min_steer, self.max_steer))
            # Also cap speed to gap-follow speed when obstacle very close
            speed_cmd = min(speed_cmd, gap_speed)

        return np.array([steer_cmd, speed_cmd])

    def get_actions_batch(self, obs, **_kwargs):
        """Get actions for every agent in the observation.

        Returns:
            actions: (num_agents, 2) array of [steering, speed]
        """
        num_agents = len(obs['poses_x'])
        actions = np.zeros((num_agents, 2))
        for i in range(num_agents):
            actions[i] = self.get_action(obs, agent_idx=i)
        return actions
