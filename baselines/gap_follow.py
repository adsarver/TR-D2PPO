import numpy as np


class GapFollow:
    """
    Reactive Gap-Following controller for F1TENTH.

    Ported from the ROS 2 node to work directly with the f110_gym observation
    dictionary, exposing the same interface as PurePursuit:

        get_action(obs, agent_idx=0)
        get_actions_batch(obs)
        update_map(map_name)
    """

    def __init__(
        self,
        map_name=None,
        num_beams=1080,
        fov=4.7,
        downsample_gap=10,
        max_sight=5.0,
        bubble_radius=8,
        extender_threshold=1.0,
        max_gap_safe_dist=1.8,
        max_speed=5.0,
        min_speed=1.0,
        max_steering=0.4189,
        **_kwargs,
    ):
        self.map_name = map_name
        self.num_beams = num_beams
        self.fov = fov
        self.downsample_gap = downsample_gap
        self.max_sight = max_sight
        self.bubble_radius = bubble_radius
        self.extender_threshold = extender_threshold
        self.max_gap_safe_dist = max_gap_safe_dist
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steering = max_steering

        # Derived constants
        self._angle_per_beam = fov / num_beams
        self._angle_per_bin = self._angle_per_beam * downsample_gap
        self._angle_offset = fov / 2.0

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample(ranges, gap):
        """Average every gap consecutive beams into one bin."""
        n_bins = len(ranges) // gap
        trimmed = ranges[: n_bins * gap]
        return trimmed.reshape(n_bins, gap).mean(axis=1)

    def _preprocess(self, raw_scan):
        """Downsample and clip a raw LiDAR scan."""
        proc = self._downsample(raw_scan, self.downsample_gap)
        np.clip(proc, 0.0, self.max_sight, out=proc)
        return proc

    # ------------------------------------------------------------------
    # Disparity extender
    # ------------------------------------------------------------------

    def _disparity_extender(self, proc):
        """Extend short-range readings around large range discontinuities
        so the planner does not try to squeeze through narrow physical gaps."""
        out = proc.copy()
        i = 0
        n = len(out)
        while i < n - 1:
            diff = out[i + 1] - out[i]
            if diff >= self.extender_threshold:
                end = min(i + self.bubble_radius + 1, n)
                out[i:end] = out[i]
                i = end
            elif -diff >= self.extender_threshold:
                start = max(0, i - self.bubble_radius + 1)
                out[start : i + 1] = out[i + 1]
                i += self.bubble_radius + 1
            else:
                i += 1
        return out

    # ------------------------------------------------------------------
    # Gap selection
    # ------------------------------------------------------------------

    def _find_max_gap(self, ranges):
        """Return (start, end) indices of the longest contiguous run of bins
        whose range exceeds max_gap_safe_dist."""
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, r in enumerate(ranges):
            if r > self.max_gap_safe_dist:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0
        return best_start, best_start + best_len

    @staticmethod
    def _best_point_in_gap(start, end, ranges):
        """Choose the midpoint of the widest gap (simple and stable)."""
        if end <= start:
            return len(ranges) // 2
        return (start + end) // 2

    @staticmethod
    def _deepest_point_in_gap(start, end, ranges):
        """Choose the farthest point inside the gap."""
        if end <= start:
            return len(ranges) // 2
        return int(start + np.argmax(ranges[start:end]))

    # ------------------------------------------------------------------
    # Steering and speed from a chosen bin index
    # ------------------------------------------------------------------

    def _bin_to_steering(self, best_bin, n_bins):
        """Convert a bin index into a steering angle.
        Bin 0 is the rightmost beam; bin n_bins-1 is the leftmost."""
        angle = best_bin * self._angle_per_bin - self._angle_offset
        return float(np.clip(angle, -self.max_steering, self.max_steering))

    def _compute_speed(self, front_min_range):
        """Smooth speed selection based on front obstacle distance.

        Uses linear interpolation between min_speed and max_speed over a
        configurable range so the car accelerates progressively as the
        path ahead clears up.
        """
        near = 1.5   # below this -> min_speed
        far  = 4.0   # above this -> max_speed
        if front_min_range <= near:
            return self.min_speed
        elif front_min_range >= far:
            return self.max_speed
        else:
            frac = (front_min_range - near) / (far - near)
            return self.min_speed + frac * (self.max_speed - self.min_speed)

    # ------------------------------------------------------------------
    # Public API  (matches PurePursuit)
    # ------------------------------------------------------------------

    def get_action(self, obs, agent_idx=0, **_kwargs):
        """Compute [steering, speed] for one agent.

        Args:
            obs:        Observation dict from f110_gym.
            agent_idx:  Which agent in a multi-agent env.

        Returns:
            np.array([steering, speed])
        """
        raw_scan = np.asarray(obs['scans'][agent_idx], dtype=np.float64)

        # Pre-process
        proc = self._preprocess(raw_scan)
        proc = self._disparity_extender(proc)

        # Find best gap
        start, end = self._find_max_gap(proc)
        best_bin = self._best_point_in_gap(start, end, proc)

        # Steering
        steering = self._bin_to_steering(best_bin, len(proc))

        # Speed: use the RAW (pre-disparity) scan in a narrow forward cone
        # (~20 deg) so wall returns at oblique angles don't dominate.
        raw_proc = self._preprocess(raw_scan)  # clipped but no disparity ext.
        n_bins = len(raw_proc)
        # Narrow cone: center 10% of bins (~27 deg out of 270 deg FOV)
        cone_half = max(1, n_bins // 20)
        center = n_bins // 2
        front_min = float(raw_proc[center - cone_half : center + cone_half].min())
        speed = self._compute_speed(front_min)

        return np.array([steering, speed])

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

    def update_map(self, map_name):
        """Accept map changes for API compatibility (gap-follow is reactive
        and does not use a map)."""
        self.map_name = map_name
