import numpy as np
from baselines.pure_pursuit import PurePursuit
from baselines.gap_follow import GapFollow


# Per-map best max_speed from speed search (FAIL maps default to 5.0)
GFPP_MAP_SPEED_LOOKUP = {
    "Austin": 5.0,
    "BrandsHatch": 11.0,
    "Budapest": 12.0,
    "Catalunya": 11.5,
    "Hockenheim": 11.0,
    "IMS": 8.5,
    "Melbourne": 11.0,
    "MexicoCity": 10.0,
    "Monza": 7.0,
    "MoscowRaceway": 13.0,
    "Nuerburgring": 11.0,
    "Oschersleben": 10.0,
    "Sakhir": 10.5,
    "SaoPaulo": 12.5,
    "Sepang": 11.0,
    "Silverstone": 9.0,
    "Sochi": 12.0,
    "Spa": 5.0,
    "Spielberg": 10.5,
    "YasMarina": 5.0,
    "Zandvoort": 10.0,
}


class GapFollowPurePursuit:
    """
    Hybrid controller that blends Pure Pursuit (raceline tracking) with
    Gap Follow (reactive obstacle avoidance) for F1TENTH.

    Ported from the ROS 2 node to work directly with the f110_gym
    observation dictionary, exposing the same interface as PurePursuit:

        get_action(obs, agent_idx=0)
        get_actions_batch(obs)
        update_map(map_name)

    Decision logic:
        - Scans front-facing LiDAR beams within a speed-dependent angular
          window (narrow at high speed, wider at low speed).
        - If the closest reading in that window drops below a dynamic
          threshold (longer at high speed), switches to Gap Follow steering
          and takes the slower of the two speed commands.
        - Otherwise follows Pure Pursuit.
        - Hysteresis prevents rapid toggling between modes.
    """

    def __init__(
        self,
        map_name,
        wheelbase=0.33,
        max_steering=0.4189,
        max_speed=12.0,
        min_speed=1.0,
        lookahead_distance=1.5,
        num_beams=1080,
        fov=4.7,
        # Obstacle detection tuning
        v_min=2.3,
        v_max=9.0,
        threshold_at_v_min=0.9,
        threshold_at_v_max=1.8,
        angle_bins_at_v_min=22,
        angle_bins_at_v_max=12,
        hysteresis=0.1,
        # GapFollow overrides
        gf_max_speed=None,
        gf_min_speed=None,
        # Optional hard speed cap (None = no extra limit)
        speed_clamp=None,
        **_kwargs,
    ):
        """
        Args:
            map_name:             Track name for PurePursuit raceline loading.
            wheelbase:            Vehicle wheelbase (m).
            max_steering:         Maximum steering angle (rad).
            max_speed / min_speed: Speed envelope passed to PurePursuit.
            lookahead_distance:   PP lookahead base distance (m).
            num_beams / fov:      LiDAR configuration.
            v_min / v_max:        Speed range over which thresholds are
                                  linearly interpolated.
            threshold_at_v_min:   Obstacle distance threshold at low speed (m).
            threshold_at_v_max:   Obstacle distance threshold at high speed (m).
            angle_bins_at_v_min:  Half-width of detection window (bins) at low speed.
            angle_bins_at_v_max:  Half-width of detection window (bins) at high speed.
            hysteresis:           Extra metres before clearing obstacle flag.
            gf_max_speed:         Override GapFollow max speed (default: min_speed*2).
            gf_min_speed:         Override GapFollow min speed (default: min_speed).
        """
        # --- Sub-controllers ---
        self.pp = PurePursuit(
            map_name=map_name,
            lookahead_distance=lookahead_distance,
            wheelbase=wheelbase,
            max_steering=max_steering,
            max_speed=max_speed,
            min_speed=min_speed,
        )

        self.gf = GapFollow(
            map_name=map_name,
            num_beams=num_beams,
            fov=fov,
            max_speed=gf_max_speed if gf_max_speed is not None else min(max_speed, min_speed * 2.5),
            min_speed=gf_min_speed if gf_min_speed is not None else min_speed,
            max_steering=max_steering,
        )

        # --- Detection parameters ---
        self.v_min = v_min
        self.v_max = v_max
        self.t_min = threshold_at_v_min
        self.t_max = threshold_at_v_max
        self.angle_min = angle_bins_at_v_max   # narrow at high speed
        self.angle_max = angle_bins_at_v_min   # wide at low speed
        self.hysteresis = hysteresis

        # LiDAR info for bin-index calculations
        self.num_beams = num_beams
        self.fov = fov
        self._downsample_gap = self.gf.downsample_gap
        self._n_bins = num_beams // self._downsample_gap

        # Per-agent obstacle state (for hysteresis)
        self._obstacle_flag = {}

        # Store for API compatibility
        self.map_name = map_name
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_clamp = speed_clamp

        # Apply per-map speed from lookup
        map_speed = GFPP_MAP_SPEED_LOOKUP.get(map_name)
        if map_speed is not None:
            self.max_speed = map_speed
            self.pp.max_speed = map_speed

    # ------------------------------------------------------------------
    # Obstacle detection
    # ------------------------------------------------------------------

    def _detect_obstacle(self, obs, agent_idx=0):
        """Return True if a nearby obstacle is detected in front of the car.

        Uses the same dynamic-threshold logic as the original ROS 2 node:
        both the distance threshold and the angular window width are linearly
        interpolated between v_min and v_max.
        """
        raw_scan = np.asarray(obs['scans'][agent_idx], dtype=np.float64)
        current_speed = float(obs['linear_vels_x'][agent_idx])

        # Clamp speed to interpolation range
        speed = np.clip(current_speed, self.v_min, self.v_max)
        frac = (speed - self.v_min) / max(self.v_max - self.v_min, 1e-6)

        # Dynamic threshold (higher at high speed = detect earlier)
        threshold = self.t_min + frac * (self.t_max - self.t_min)

        # Dynamic angular half-width in raw beams (narrower at high speed)
        half_angle_bins = int(self.angle_max + frac * (self.angle_min - self.angle_max))

        # Front-facing window in the raw scan
        center = len(raw_scan) // 2
        start = max(0, center - half_angle_bins)
        end = min(len(raw_scan), center + half_angle_bins)
        front = raw_scan[start:end]

        if len(front) == 0:
            return False

        min_dist = float(np.min(front))

        # Hysteresis
        was_detected = self._obstacle_flag.get(agent_idx, False)
        if min_dist < threshold:
            detected = True
        elif min_dist > threshold + self.hysteresis:
            detected = False
        else:
            detected = was_detected  # stay in previous state

        self._obstacle_flag[agent_idx] = detected
        return detected

    # ------------------------------------------------------------------
    # Public API  (matches PurePursuit)
    # ------------------------------------------------------------------

    def get_action(self, obs, agent_idx=0, **_kwargs):
        """Compute [steering, speed] for one agent.

        If an obstacle is detected: use GapFollow steering, min of both speeds.
        Otherwise: use PurePursuit.
        """
        pp_action = self.pp.get_action(obs, agent_idx=agent_idx)
        obstacle = self._detect_obstacle(obs, agent_idx=agent_idx)

        if obstacle:
            gf_action = self.gf.get_action(obs, agent_idx=agent_idx)
            steering = gf_action[0]
            speed = min(pp_action[1], gf_action[1])
            action = np.array([steering, speed])
        else:
            action = pp_action

        if self.speed_clamp is not None:
            action[1] = min(action[1], self.speed_clamp)
        return action

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
        """Switch both sub-controllers to a new map."""
        self.map_name = map_name
        self.pp.update_map(map_name)
        self.gf.update_map(map_name)
        self._obstacle_flag.clear()

        # Apply per-map speed from lookup
        map_speed = GFPP_MAP_SPEED_LOOKUP.get(map_name)
        if map_speed is not None:
            self.max_speed = map_speed
            self.pp.max_speed = map_speed
