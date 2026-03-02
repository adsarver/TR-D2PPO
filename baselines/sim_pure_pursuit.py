import numpy as np
from scipy.spatial import distance, transform

from utils.utils import get_map_dir

class SimPurePursuit:
    """
    Simulation-compatible Pure Pursuit controller (no ROS).
    Usage:
        action = SimPurePursuit().get_action(curr_pose, waypoints, ref_speed)
    """
    def __init__(self, map_name='BrandsHatch'):
        self.map_name = map_name
        self.is_ascending = True
        self.numWaypoints = None
        self.waypoints = None
        self.ref_speed = None
        
    def update_map(self, map_name):
        self.map_name = map_name
        csv_path = get_map_dir(map_name) + f"/{map_name}_raceline.csv"
        csv_data = np.loadtxt(csv_path, delimiter=';', skiprows=1)
        self.load_waypoints(csv_data, xind=1, yind=2, vind=5)

    def load_waypoints(self, csv_data, xind, yind, vind):
        self.waypoints = csv_data[:, [xind, yind]]
        self.numWaypoints = self.waypoints.shape[0]
        self.ref_speed = csv_data[:, vind] * 0.9

    def build_control_functions(self, speed, v_min, v_max):
        d_min = 0.8
        d_max = 4.0
        m_L = (d_max - d_min) / (v_max - v_min)
        b_L = d_min - m_L * v_min
        L = m_L * speed + b_L

        gain_max = 0.9
        gain_min = 0.65
        m_gain = (gain_min - gain_max) / (v_max - v_min)
        b_gain = gain_max - m_gain * v_min
        steering_gain = m_gain * speed + b_gain

        return L, steering_gain

    def get_actions_batch(self, obs, waypoints=None, ref_speed=None):
        # curr_pose: [x, y, theta, qx, qy, qz, qw] (theta optional)
        # waypoints: [N, 2] (optional, uses self.waypoints if None)
        # ref_speed: [N] (optional, uses self.ref_speed if None)
        if waypoints is not None:
            self.waypoints = waypoints
            self.numWaypoints = waypoints.shape[0]
        if ref_speed is not None:
            self.ref_speed = ref_speed

        curr_pose = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        
        currX, currY = curr_pose[0], curr_pose[1]
        currPos = np.array([currX, currY]).reshape((1, 2))
        # Since your state space only has x, y, and theta, construct a quaternion from theta (yaw)
        theta = curr_pose[2]
        # Quaternion for yaw-only rotation: [x, y, z, w]
        quat = [0, 0, np.sin(theta / 2), np.cos(theta / 2)]
        R = transform.Rotation.from_quat(quat)
        rot = R.as_matrix()

        distances = distance.cdist(currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        closest_index = np.argmin(distances)
        closestPoint = self.waypoints[closest_index]
        speed = self.ref_speed[closest_index]
        v_min = 3.78
        v_max = 9.0
        L, steering_gain = self.build_control_functions(speed, v_min, v_max)
        if speed >= 6.3:
            L += 1.0
        targetPoint = self.get_closest_point_beyond_lookahead_dist(L, distances, closest_index)
        translatedTargetPoint = self.translatePoint(targetPoint, currPos, rot, currX, currY)
        y = translatedTargetPoint[1]
        gamma = steering_gain * (2 * y / L**2)
        gamma = np.clip(gamma, -0.35, 0.35)
        action_speed = speed
        return np.array([[gamma, action_speed]])

    def get_closest_point_beyond_lookahead_dist(self, threshold, distances, closest_index):
        point_index = closest_index
        dist = distances[point_index]
        while dist < threshold:
            if self.is_ascending:
                point_index += 1
                if point_index >= self.numWaypoints:
                    point_index = 0
                dist = distances[point_index]
            else:
                point_index -= 1
                if point_index < 0:
                    point_index = self.numWaypoints - 1
                dist = distances[point_index]
        return self.waypoints[point_index]

    def translatePoint(self, targetPoint, currPos, rot, currX, currY):
        H = np.zeros((4, 4))
        H[0:3, 0:3] = np.linalg.inv(rot)
        H[0, 3] = currX
        H[1, 3] = currY
        H[3, 3] = 1.0
        pvect = targetPoint - currPos
        convertedTarget = (H @ np.array((pvect[0, 0], pvect[0, 1], 0, 0))).reshape((4))
        return convertedTarget
