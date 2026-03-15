"""RescueMissionAgent — Sequential mission planner for autonomous victim rescue.

Weighs proximity and audio urgency to decide victim priority, generates a
discrete task plan (TURN_TO, WALK, GRAB, RESCUE), and executes one task at a
time via env actions. Replans after each rescue.

Actions mapping:
    0 = MOVE_FORWARD
    1 = TURN_LEFT
    2 = TURN_RIGHT
    3 = LIFT_DEBRIS
    4 = RESCUE_PERSON
    5 = STEP_OVER
"""

import math
import numpy as np


def _angle_diff(target, current):
    """Signed angular difference in [-pi, pi]."""
    d = target - current
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _bearing_deg(robot_xy, robot_yaw, target_xy):
    """Signed bearing in degrees from robot heading to target."""
    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    angle_to = math.atan2(dy, dx)
    diff = _angle_diff(angle_to, robot_yaw)
    return math.degrees(diff)


class RescueMissionAgent:
    """Sequential mission planner that prioritises victims by proximity and
    audio urgency, then generates an ordered task list of discrete steps."""

    # Scoring weights
    URGENCY_WEIGHT = 1.0
    PROXIMITY_WEIGHT = 1.0

    # Task completion thresholds
    HEADING_TOLERANCE = 0.1      # rad — close enough when turning
    APPROACH_DISTANCE = 0.9      # m — must be within VICTIM_RESCUE_DIST (1.0m)

    def __init__(self, victim_positions, victim_debris):
        """
        Args:
            victim_positions: list of np.array([x, y]) for each victim.
            victim_debris: dict  victim_id -> list of debris body name strings.
        """
        self.victim_positions = victim_positions
        self.victim_debris = victim_debris
        self._executed = set()  # track which single-shot tasks have been executed

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_victim(self, robot_xy, victim_idx):
        """Score = urgency_weight / (d+0.1) + proximity_weight / (d+0.1).

        Both terms use the same distance — closer victims are louder AND
        nearer, so they naturally dominate.
        """
        vpos = self.victim_positions[victim_idx]
        d = float(np.linalg.norm(robot_xy - vpos))
        inv_d = 1.0 / (d + 0.1)
        return self.URGENCY_WEIGHT * inv_d + self.PROXIMITY_WEIGHT * inv_d

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _priority_order(self, robot_xy, victims_rescued):
        """Return victim indices sorted by descending score (best first),
        skipping already-rescued victims."""
        remaining = [i for i in range(len(self.victim_positions))
                     if i not in victims_rescued]
        scored = [(self._score_victim(robot_xy, i), i) for i in remaining]
        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def plan(self, robot_xy, robot_yaw, victims_rescued, debris_cleared):
        """Generate an ordered task list for the current state.

        Returns a list of task dicts.  Each dict has at least:
            type:  str  — TURN_TO | WALK | GRAB | RESCUE
        Plus type-specific keys (target_xy, victim_idx, debris_name, ...).
        """
        self._executed = set()  # reset on replan
        order = self._priority_order(robot_xy, victims_rescued)
        if not order:
            return []

        tasks = []
        current_xy = np.array(robot_xy, dtype=float)
        current_yaw = float(robot_yaw)

        for vidx in order:
            vpos = self.victim_positions[vidx]
            dist = float(np.linalg.norm(current_xy - vpos))
            bearing = _bearing_deg(current_xy, current_yaw, vpos)
            target_yaw = math.atan2(vpos[1] - current_xy[1],
                                    vpos[0] - current_xy[0])

            # 1. Turn toward victim
            tasks.append({
                "type": "TURN_TO",
                "victim_idx": vidx,
                "target_xy": vpos.copy(),
                "bearing_deg": bearing,
                "target_yaw": target_yaw,
            })

            # 2. Walk to victim
            tasks.append({
                "type": "WALK",
                "victim_idx": vidx,
                "target_xy": vpos.copy(),
                "distance": dist,
            })

            # 3. Grab each piece of debris (skip already cleared), settle after each
            cleared = debris_cleared.get(vidx, set())
            for dname in self.victim_debris.get(vidx, []):
                if dname not in cleared:
                    tasks.append({
                        "type": "GRAB",
                        "victim_idx": vidx,
                        "debris_name": dname,
                    })
                    # Settle: stand still to recover balance after grab
                    tasks.append({
                        "type": "SETTLE",
                        "victim_idx": vidx,
                        "settle_steps": 10,
                    })

            # 4. Rescue
            tasks.append({
                "type": "RESCUE",
                "victim_idx": vidx,
            })

            # Update simulated position for next victim's bearing/distance
            current_xy = vpos.copy()
            current_yaw = target_yaw

        return tasks

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def get_next_action(self, robot_xy, robot_yaw, current_task):
        """Given the current task, return the env action int (0-5) to execute.

        TURN_TO  -> 1 (TURN_LEFT) or 2 (TURN_RIGHT)
        WALK     -> 0 (MOVE_FORWARD), with turning correction if needed
        GRAB     -> 3 (LIFT_DEBRIS)
        RESCUE   -> 4 (RESCUE_PERSON)
        """
        ttype = current_task["type"]

        if ttype == "TURN_TO":
            target_yaw = current_task["target_yaw"]
            diff = _angle_diff(target_yaw, robot_yaw)
            if diff > 0:
                return 1   # TURN_LEFT
            else:
                return 2   # TURN_RIGHT

        if ttype == "WALK":
            target_xy = current_task["target_xy"]
            dx = target_xy[0] - robot_xy[0]
            dy = target_xy[1] - robot_xy[1]
            angle_to = math.atan2(dy, dx)
            diff = _angle_diff(angle_to, robot_yaw)
            # Correct heading if drifting
            if abs(diff) > 0.25:
                return 1 if diff > 0 else 2
            return 0   # MOVE_FORWARD

        if ttype == "GRAB":
            self._executed.add(id(current_task))
            return 3   # LIFT_DEBRIS (crouch-grab-throw)

        if ttype == "RESCUE":
            self._executed.add(id(current_task))
            return 4   # RESCUE_PERSON

        if ttype == "SETTLE":
            return 4   # RESCUE_PERSON does nothing physical — just stands still

        return 0  # fallback

    def is_task_complete(self, robot_xy, robot_yaw, current_task):
        """Check whether the current task's completion condition is met."""
        ttype = current_task["type"]

        if ttype == "TURN_TO":
            target_yaw = current_task["target_yaw"]
            diff = abs(_angle_diff(target_yaw, robot_yaw))
            return diff < self.HEADING_TOLERANCE

        if ttype == "WALK":
            target_xy = current_task["target_xy"]
            dist = float(np.linalg.norm(
                np.array(robot_xy) - np.array(target_xy)))
            return dist < self.APPROACH_DISTANCE

        # GRAB and RESCUE: complete after the action has been executed once
        if ttype in ("GRAB", "RESCUE"):
            task_id = id(current_task)
            if task_id in self._executed:
                return True
            return False

        # SETTLE: complete after N action steps
        if ttype == "SETTLE":
            remaining = current_task.get("settle_steps", 5)
            if remaining <= 0:
                return True
            current_task["settle_steps"] = remaining - 1
            return False

        return True

    # ------------------------------------------------------------------
    # Audio urgency helper (for console output)
    # ------------------------------------------------------------------

    @staticmethod
    def audio_urgency(distance):
        """Return an audio urgency string based on distance to victim."""
        if distance < 1.5:
            return 'CRITICAL -- "I\'m right here!"'
        if distance < 3.0:
            return '"Help! Can you hear me?"'
        if distance < 5.0:
            return '"Is anyone there?"'
        return None
