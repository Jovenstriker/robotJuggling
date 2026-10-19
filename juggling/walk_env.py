"""Gymnasium environment: a free-standing planar robot that juggles a soccer ball
and steps / hops to stay under it (juggler_walk.xml).

The torso moves in the x-z plane (forward, up, pitch) and must be balanced on the
feet. The ball also moves in that plane and is launched with a random forward or
backward speed, so standing still is not enough: the robot only moves to keep
juggling. There is no reward for walking itself.

Action (6,) in [-1, 1]: joint position targets, offset from a home pose that
                        stands on the left leg with the right foot raised.
Obs    (33,): torso height / pitch / velocities, joint pos & vel, ball pos & vel
              relative to the torso, ball rel. each foot, foot velocities,
              foot pitch, foot-on-ground flags.
Episode ends when the ball drops, gets too far from the robot, the robot falls,
or after `max_steps` control steps (20 s at 50 Hz).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from juggling.env import BALL_RADIUS, MIN_JUGGLE_RISE

XML_PATH = Path(__file__).parent / "assets" / "juggler_walk.xml"
LEG_JOINTS = ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"]
# stand on the left leg, right foot raised and flat (same as the stationary juggler)
HOME_DEG = np.array([-75.0, 35.0, 40.0, 0.0, 5.0, -5.0])
ACTION_SCALE_DEG = np.array([50.0, 60.0, 45.0, 60.0, 60.0, 40.0])


class SoccerJuggleWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        frame_skip=10,
        max_steps=1000,
        drop_height=0.25,
        max_ball_dist=0.7,
        target_apex=1.05,
        ball_speed=0.3,
        alternate_bonus=0.5,
        width=640,
        height=480,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.drop_height = drop_height
        self.max_ball_dist = max_ball_dist
        self.target_apex = target_apex
        self.ball_speed = ball_speed  # max |vx| of the ball at launch (m/s)
        self.alternate_bonus = alternate_bonus
        self.render_mode = render_mode
        self._width, self._height = width, height
        self._renderer = None
        self._viewer = None

        m = self.model
        self._leg_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in LEG_JOINTS])
        self._leg_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS])
        self._root_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in ["root_x", "root_z", "root_pitch"]])
        self._root_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in ["root_x", "root_z", "root_pitch"]])
        self._ball_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in ["ball_x", "ball_z", "ball_rot"]])
        self._ball_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in ["ball_x", "ball_z", "ball_rot"]])
        self._ball_body = m.body("ball").id
        self._torso_body = m.body("torso").id
        self._ball_geom = m.geom("ball").id
        self._floor_geom = m.geom("floor").id
        self._foot_geoms = [m.geom("r_foot").id, m.geom("l_foot").id]
        self._robot_geoms = {m.geom(f"{s}_{g}").id for s in "rl" for g in ["foot", "shin", "thigh"]}
        self._foot_sites = [m.site("r_foot_top").id, m.site("l_foot_top").id]
        self._foot_bodies = [m.body("r_foot").id, m.body("l_foot").id]

        self._home = np.radians(HOME_DEG)
        self._scale = np.radians(ACTION_SCALE_DEG)
        self._ctrl_lo, self._ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self._torso_z0 = m.body_pos[self._torso_body][2]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    def _torso(self):
        x, z = self.data.xpos[self._torso_body][[0, 2]]
        pitch = self.data.qpos[self._root_qpos[2]]
        return x, z, pitch

    def _ball(self):
        p = self.data.xpos[self._ball_body][[0, 2]].copy()
        v = self.data.qvel[self._ball_qvel[:2]].copy()
        return p, v

    def _feet(self):
        """(2, 2) foot-top x-z positions, (2, 2) x-z velocities, (2,) foot pitch."""
        pos, vel, pitch = [], [], []
        v6 = np.zeros(6)
        for s, b in zip(self._foot_sites, self._foot_bodies):
            pos.append(self.data.site_xpos[s][[0, 2]].copy())
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, s, v6, 0)
            vel.append(v6[[3, 5]].copy())
            xmat = self.data.xmat[b].reshape(3, 3)
            pitch.append(np.arctan2(-xmat[2, 0], xmat[0, 0]))
        return np.array(pos), np.array(vel), np.array(pitch)

    def _contacts(self):
        """Ball contacts (per foot, other leg geoms, floor) and feet-on-ground flags."""
        ball_feet = [False, False]
        ball_leg = ball_floor = False
        grounded = [False, False]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._ball_geom in pair:
                other = (pair - {self._ball_geom}).pop()
                if other in self._foot_geoms:
                    ball_feet[self._foot_geoms.index(other)] = True
                elif other in self._robot_geoms:
                    ball_leg = True
                ball_floor |= other == self._floor_geom
            elif self._floor_geom in pair:
                other = (pair - {self._floor_geom}).pop()
                if other in self._foot_geoms:
                    grounded[self._foot_geoms.index(other)] = True
        return ball_feet, ball_leg, ball_floor, grounded

    def _obs(self):
        tx, tz, pitch = self._torso()
        ball_p, ball_v = self._ball()
        feet_p, feet_v, feet_pitch = self._feet()
        return np.concatenate(
            [
                [tz - self._torso_z0, pitch],
                self.data.qvel[self._root_qvel] * 0.2,
                self.data.qpos[self._leg_qpos],
                self.data.qvel[self._leg_qvel] * 0.1,
                ball_p - [tx, tz],
                ball_v * 0.2,
                (ball_p - feet_p).ravel(),
                feet_v.ravel() * 0.2,
                feet_pitch,
                np.asarray(self._grounded, dtype=np.float64),
            ]
        )

    # --------------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        q = self._home + rng.uniform(-0.03, 0.03, size=6)
        self.data.qpos[self._leg_qpos] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        # put the standing (left) foot's sole exactly on the ground
        sole = self.data.geom_xpos[self._foot_geoms[1]][2] - 0.025
        self.data.qpos[self._root_qpos[1]] -= sole
        mujoco.mj_forward(self.model, self.data)

        # drop the ball above the raised foot, with a random horizontal speed
        foot = self.data.site_xpos[self._foot_sites[0]]
        # where the raised foot sits relative to the torso: the robot-relative juggling spot
        self._juggle_offset = foot[0] - self._torso()[0]
        self.data.qpos[self._ball_qpos[0]] = foot[0] + rng.uniform(-0.05, 0.05) - self.model.body_pos[self._ball_body][0]
        self.data.qpos[self._ball_qpos[1]] = foot[2] + rng.uniform(0.5, 0.8) - self.model.body_pos[self._ball_body][2]
        self.data.qvel[self._ball_qvel[0]] = rng.uniform(-self.ball_speed, self.ball_speed)
        self.data.qvel[self._ball_qvel[1]] = rng.uniform(-0.5, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._touches = 0
        self._foot_touches = [0, 0]
        self._was_touching = [False, False]
        self._grounded = [False, True]
        self._touch_z = None
        self._touch_foot = None
        self._last_juggle_foot = None
        self._start_x = self._torso()[0]
        self._prev_action = np.zeros(6)
        return self._obs(), self._info()

    def _info(self):
        tx = self._torso()[0]
        return {
            "touches": self._touches,
            "foot_touches": list(self._foot_touches),
            "distance": tx - self._start_x,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = np.clip(self._home + action * self._scale, self._ctrl_lo, self._ctrl_hi)

        new_touch_foot = None
        floor_hit = leg_hit = False
        contact_substeps = 0
        grounded = [False, False]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            feet, leg, floor, on_ground = self._contacts()
            for k in range(2):
                if feet[k] and not self._was_touching[k]:
                    new_touch_foot = k
                grounded[k] |= on_ground[k]
            self._was_touching = feet
            contact_substeps += any(feet) or leg
            leg_hit |= leg
            floor_hit |= floor
        self._grounded = grounded
        self._steps += 1

        tx, tz, pitch = self._torso()
        ball_p, ball_v = self._ball()
        feet_p, _, _ = self._feet()
        # the ball should stay in front of the robot, where a foot can reach it
        ball_err = abs(ball_p[0] - (tx + self._juggle_offset))
        feet_x_err = np.abs(ball_p[0] - feet_p[:, 0])
        nearest = int(np.argmin(feet_x_err))

        # ---- reward ----
        reward = 0.2  # upright and still playing
        reward -= 0.5 * abs(pitch)  # keep the torso upright
        if contact_substeps == 0:
            gap = ball_p[1] - feet_p[nearest, 1] - BALL_RADIUS
            reward += 0.5 + 0.5 * np.clip(gap / 0.4, 0.0, 1.0)
        reward -= 0.05 * contact_substeps
        reward -= 0.5 * leg_hit
        reward -= 1.0 * max(ball_err - 0.25, 0.0)  # free zone: robot need not be exact
        if ball_v[1] < 0:
            reward -= 0.5 * feet_x_err[nearest]
        if new_touch_foot is not None:
            self._touch_z = ball_p[1]
            self._touch_foot = new_touch_foot
        if self._touch_z is not None and ball_v[1] <= 0 and contact_substeps == 0:
            if ball_p[1] - self._touch_z > MIN_JUGGLE_RISE:
                self._touches += 1
                self._foot_touches[self._touch_foot] += 1
                apex_err = ball_p[1] - self.target_apex
                reward += 2.0 + 4.0 * np.exp(-((apex_err / 0.2) ** 2))
                if self._last_juggle_foot is not None and self._touch_foot != self._last_juggle_foot:
                    reward += self.alternate_bonus
                self._last_juggle_foot = self._touch_foot
            self._touch_z = None
        reward -= 0.05 * float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action

        fell = tz < 0.6 or abs(pitch) > 0.8
        dropped = ball_p[1] < self.drop_height or floor_hit
        lost = abs(ball_p[0] - tx) > self._juggle_offset + self.max_ball_dist
        terminated = bool(fell or dropped or lost)
        if terminated:
            reward -= 10.0
        truncated = self._steps >= self.max_steps

        info = self._info()
        info.update(fell=fell, dropped=dropped)
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, self._height, self._width)
                self._cam = mujoco.MjvCamera()
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 3.0, 110, -10
            self._cam.lookat[:] = [self._torso()[0] + 0.3, 0.0, 0.6]  # follow the robot
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer  # on macOS, run the script with `mjpython`

            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                self._viewer.cam.trackbodyid = self._torso_body
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


gym.register(id="SoccerJuggleWalk-v0", entry_point="juggling.walk_env:SoccerJuggleWalkEnv")
