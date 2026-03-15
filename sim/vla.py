"""RescueVLA — Lightweight Vision-Language-Action system using MuJoCo raycasting.

No external models or API calls. Perceives the environment via raycasting,
detects obstacles and victims, and makes rule-based action decisions for
autonomous rescue operations.
"""

import math
import numpy as np
import mujoco

from sim.rewards import VICTIM_POSITIONS, VICTIM_DEBRIS


# Audio detection range (meters) — robot "hears" victim calls
_AUDIO_DETECT_RANGE = 5.0

# Ray configuration
_NUM_RAYS = 7
_FAN_HALF_ANGLE = math.radians(30)  # +/- 30 degrees
_RAY_HEIGHTS = [0.05, 0.15, 0.30]   # cast at multiple heights above ground
_RAY_MAX_DIST = 6.0                  # max raycast distance

# Obstacle height thresholds
_STEP_OVER_THRESHOLD = 0.15   # obstacles below this can be stepped over
_OBSTACLE_THRESHOLD = 0.02    # anything above this is an obstacle

# Proximity thresholds
_VICTIM_NEARBY_DIST = 1.8     # close enough to interact with victim
_VICTIM_APPROACH_DIST = 6.0   # far enough to navigate toward


class RescueVLA:
    """Rule-based Vision-Language-Action system for autonomous rescue.

    Uses MuJoCo raycasting to perceive obstacles and victims, then
    applies deterministic rules to select the best action.
    """

    def __init__(self, model, data, fj_qpos_adr):
        self.model = model
        self.data = data
        self._fj_qpos_adr = fj_qpos_adr

        # Cache victim body IDs
        self._victim_body_ids = {}
        self._victim_geom_ids = {}
        for i in range(1, 5):
            body_name = f"victim_{i}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                self._victim_body_ids[i - 1] = bid
                # Cache geom IDs for this victim body
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] == bid:
                        self._victim_geom_ids[gid] = i - 1  # geom_id -> victim_index

        # Cache debris geom IDs
        self._debris_geom_ids = {}
        for victim_id, debris_names in VICTIM_DEBRIS.items():
            for dname in debris_names:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, dname)
                if bid >= 0:
                    for gid in range(model.ngeom):
                        if model.geom_bodyid[gid] == bid:
                            self._debris_geom_ids[gid] = (victim_id, dname)

        # Ground body/geom IDs (to exclude from obstacle detection)
        self._ground_body_id = 0  # worldbody is always 0

        # Robot body IDs (to exclude from detection)
        self._robot_body_ids = set()
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id >= 0:
            self._collect_child_bodies(pelvis_id)

        # State tracking
        self._last_decision = "WALK_FORWARD"
        self._stuck_counter = 0
        self._last_pos = None
        self._navigate_around_dir = 1  # 1=left, -1=right, alternates

    def _collect_child_bodies(self, body_id):
        """Recursively collect all body IDs in the robot kinematic tree."""
        self._robot_body_ids.add(body_id)
        for b in range(self.model.nbody):
            if self.model.body_parentid[b] == body_id and b != body_id:
                self._collect_child_bodies(b)

    def _get_robot_state(self, data):
        """Get robot position and yaw from free joint."""
        a = self._fj_qpos_adr
        pos = data.qpos[a:a + 3].copy()
        q = data.qpos[a + 3:a + 7]
        w, x, y, z = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos, yaw

    def perceive(self, data):
        """Cast rays in a fan pattern to detect obstacles and victims.

        Returns a perception dict with:
            - obstacles: list of (angle, distance, height) for detected obstacles
            - victims_detected: list of (victim_index, angle, distance)
            - debris_detected: list of (victim_id, debris_name, angle, distance)
            - clear_ahead: bool — no obstacles in the center rays
            - nearest_obstacle_dist: float
            - nearest_obstacle_height: float
            - audio_victims: list of victim indices within audio range
            - summary: str — human-readable summary
        """
        pos, yaw = self._get_robot_state(data)
        robot_xy = pos[:2]

        obstacles = []
        victims_detected = []
        debris_detected = []

        # Cast rays in a fan pattern
        ray_angles = np.linspace(-_FAN_HALF_ANGLE, _FAN_HALF_ANGLE, _NUM_RAYS)

        for ray_idx, angle_offset in enumerate(ray_angles):
            ray_yaw = yaw + angle_offset

            for height in _RAY_HEIGHTS:
                # Ray origin: robot position at the specified height
                ray_origin = np.array([pos[0], pos[1], height + 0.02], dtype=np.float64)

                # Ray direction: forward in the robot's local frame
                ray_dir = np.array([
                    math.cos(ray_yaw),
                    math.sin(ray_yaw),
                    0.0
                ], dtype=np.float64)

                # Cast the ray
                geom_id = np.array([-1], dtype=np.int32)
                dist = mujoco.mj_ray(
                    self.model, data,
                    ray_origin, ray_dir,
                    None,  # geomgroup filter (None = all)
                    1,     # flg_static
                    -1,    # bodyexclude (-1 = none)
                    geom_id
                )

                if dist < 0 or dist > _RAY_MAX_DIST:
                    continue

                gid = int(geom_id[0])
                if gid < 0:
                    continue

                # Check what we hit
                hit_body = self.model.geom_bodyid[gid]

                # Skip robot's own body
                if hit_body in self._robot_body_ids:
                    continue

                # Check if it's a victim
                if gid in self._victim_geom_ids:
                    victim_idx = self._victim_geom_ids[gid]
                    victims_detected.append((victim_idx, angle_offset, dist))
                    continue

                # Check if it's victim debris
                if gid in self._debris_geom_ids:
                    vid, dname = self._debris_geom_ids[gid]
                    debris_detected.append((vid, dname, angle_offset, dist))
                    # Also treat as obstacle
                    obstacles.append((angle_offset, dist, height))
                    continue

                # Skip ground plane (body 0, and geoms at z ~ 0)
                if hit_body == self._ground_body_id:
                    continue

                # It's a scene obstacle
                obstacles.append((angle_offset, dist, height))

        # Audio detection — check distance to each victim
        audio_victims = []
        for i, vpos in enumerate(VICTIM_POSITIONS):
            d = np.linalg.norm(robot_xy - vpos)
            if d < _AUDIO_DETECT_RANGE:
                audio_victims.append((i, d))

        # Analyze center rays for clear path
        center_obstacles = [o for o in obstacles if abs(o[0]) < math.radians(12)]
        nearest_obstacle_dist = min((o[1] for o in center_obstacles), default=_RAY_MAX_DIST)
        nearest_obstacle_height = max((o[2] for o in center_obstacles), default=0.0)
        clear_ahead = nearest_obstacle_dist > 2.0

        # Deduplicate victims by index
        seen_victims = {}
        for vidx, ang, d in victims_detected:
            if vidx not in seen_victims or d < seen_victims[vidx][1]:
                seen_victims[vidx] = (ang, d)
        victims_detected_dedup = [(vidx, ang, d) for vidx, (ang, d) in seen_victims.items()]

        # Build summary
        parts = []
        if victims_detected_dedup:
            vnames = [f"V{v[0]+1}@{v[2]:.1f}m" for v in victims_detected_dedup]
            parts.append(f"Victims: {', '.join(vnames)}")
        if debris_detected:
            parts.append(f"Debris: {len(set(d[1] for d in debris_detected))} pieces")
        if center_obstacles:
            parts.append(f"Obstacle@{nearest_obstacle_dist:.1f}m h={nearest_obstacle_height:.2f}m")
        if audio_victims:
            anames = [f"V{a[0]+1}@{a[1]:.1f}m" for a in audio_victims]
            parts.append(f"Audio: {', '.join(anames)}")
        if not parts:
            parts.append("Clear path")
        summary = " | ".join(parts)

        return {
            "obstacles": obstacles,
            "victims_detected": victims_detected_dedup,
            "debris_detected": debris_detected,
            "clear_ahead": clear_ahead,
            "nearest_obstacle_dist": nearest_obstacle_dist,
            "nearest_obstacle_height": nearest_obstacle_height,
            "audio_victims": audio_victims,
            "summary": summary,
        }

    def decide(self, perception, robot_pos, robot_yaw,
               victims_reached, victims_rescued, debris_cleared):
        """Decide the next action based on perception and mission state.

        Args:
            perception: dict from perceive()
            robot_pos: [x, y, z] robot position
            robot_yaw: float, robot heading in radians
            victims_reached: set of victim indices reached
            victims_rescued: set of victim indices rescued
            debris_cleared: dict of {victim_id: set of cleared debris names}

        Returns:
            decision: str — action name
            target_info: dict with optional target direction/position info
        """
        robot_xy = robot_pos[:2]

        # Find nearest unrescued victim
        nearest_victim = None
        nearest_dist = float("inf")
        for i, vpos in enumerate(VICTIM_POSITIONS):
            if i in victims_rescued:
                continue
            d = np.linalg.norm(robot_xy - vpos)
            if d < nearest_dist:
                nearest_dist = d
                nearest_victim = i

        if nearest_victim is None:
            return "WALK_FORWARD", {"reason": "All victims rescued"}

        target_pos = VICTIM_POSITIONS[nearest_victim]
        target_dist = nearest_dist

        # Angle to target
        dx = target_pos[0] - robot_xy[0]
        dy = target_pos[1] - robot_xy[1]
        angle_to_target = math.atan2(dy, dx) - robot_yaw
        while angle_to_target > math.pi:
            angle_to_target -= 2 * math.pi
        while angle_to_target < -math.pi:
            angle_to_target += 2 * math.pi

        # Check if victim is nearby
        victim_nearby = target_dist < _VICTIM_NEARBY_DIST

        # Check if debris is cleared for this victim
        required_debris = VICTIM_DEBRIS.get(nearest_victim, [])
        cleared_debris = debris_cleared.get(nearest_victim, set())
        all_debris_cleared = len(cleared_debris) >= len(required_debris)

        # Priority 1: If victim nearby and debris not cleared -> CROUCH_GRAB
        if victim_nearby and not all_debris_cleared:
            return "CROUCH_GRAB", {
                "reason": f"Clearing debris for V{nearest_victim+1}",
                "victim": nearest_victim,
                "target_angle": angle_to_target,
            }

        # Priority 2: If victim nearby and debris cleared -> RESCUE
        if victim_nearby and all_debris_cleared:
            return "RESCUE", {
                "reason": f"Rescuing V{nearest_victim+1}",
                "victim": nearest_victim,
                "target_angle": angle_to_target,
            }

        # Priority 3: Check for obstacles ahead
        if not perception["clear_ahead"]:
            obs_dist = perception["nearest_obstacle_dist"]
            obs_height = perception["nearest_obstacle_height"]

            if obs_dist < 1.5:
                # Obstacle is close
                if obs_height <= _STEP_OVER_THRESHOLD:
                    return "STEP_OVER", {
                        "reason": f"Stepping over obstacle h={obs_height:.2f}m",
                        "obstacle_dist": obs_dist,
                        "obstacle_height": obs_height,
                        "target_angle": angle_to_target,
                    }
                else:
                    # Tall obstacle — navigate around
                    # Choose direction: prefer the side closer to target
                    if angle_to_target > 0:
                        self._navigate_around_dir = 1  # go left
                    else:
                        self._navigate_around_dir = -1  # go right
                    return "NAVIGATE_AROUND", {
                        "reason": f"Avoiding tall obstacle h={obs_height:.2f}m",
                        "direction": "LEFT" if self._navigate_around_dir > 0 else "RIGHT",
                        "obstacle_dist": obs_dist,
                        "target_angle": angle_to_target,
                    }

        # Priority 4: Navigate toward nearest victim
        # Detect stuck condition
        if self._last_pos is not None:
            moved = np.linalg.norm(robot_xy - self._last_pos)
            if moved < 0.02:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._last_pos = robot_xy.copy()

        if self._stuck_counter > 10:
            self._stuck_counter = 0
            self._navigate_around_dir *= -1
            return "NAVIGATE_AROUND", {
                "reason": "Stuck — trying alternate path",
                "direction": "LEFT" if self._navigate_around_dir > 0 else "RIGHT",
                "target_angle": angle_to_target,
            }

        # Need to turn toward target?
        if abs(angle_to_target) > 0.15:
            return "WALK_TOWARD", {
                "reason": f"Heading to V{nearest_victim+1} ({target_dist:.1f}m)",
                "victim": nearest_victim,
                "target_angle": angle_to_target,
                "target_dist": target_dist,
            }

        # Default: walk forward toward target
        return "WALK_FORWARD", {
            "reason": f"Walking to V{nearest_victim+1} ({target_dist:.1f}m)",
            "victim": nearest_victim,
            "target_dist": target_dist,
        }

    def plan_route(self, robot_xy, victims_rescued):
        """Plan visit order for unrescued victims (nearest-first greedy).

        Returns list of victim indices in planned visit order.
        """
        remaining = [i for i in range(len(VICTIM_POSITIONS)) if i not in victims_rescued]
        route = []
        current = robot_xy.copy()
        while remaining:
            dists = [(np.linalg.norm(current - VICTIM_POSITIONS[i]), i) for i in remaining]
            dists.sort()
            nearest_idx = dists[0][1]
            route.append(nearest_idx)
            current = VICTIM_POSITIONS[nearest_idx].copy()
            remaining.remove(nearest_idx)
        return route
