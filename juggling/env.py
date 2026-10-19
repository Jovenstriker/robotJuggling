"""Gymnasium environment: a fixed-base robot leg juggling a soccer ball in MuJoCo.

Action  (5,)  in [-1, 1]: joint position targets, offset from a "home" pose
              (hip pitch, hip roll, knee, ankle pitch, ankle roll).
Obs     (25,): leg qpos (5), leg qvel (5), ball pos rel. foot (3), ball vel (3),
               foot normal (3), foot pos rel. home (3), foot linear vel (3).
Reward favors an airborne ball, real bounces off the foot (counted as "touches"
in info) with an apex near `target_apex`, and penalizes cradling the ball.
Episode ends when the ball drops below `drop_height`, drifts too far, or after
`max_steps` control steps (20 s at 50 Hz by default).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML_PATH = Path(__file__).parent / "assets" / "juggler.xml"
LEG_JOINTS = ["hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]
HOME_DEG = np.array([-75.0, 0.0, 35.0, 40.0, 0.0])
# how far (deg) the policy may move each joint target away from home
ACTION_SCALE_DEG = np.array([45.0, 25.0, 60.0, 40.0, 25.0])
BALL_RADIUS = 0.11
MIN_JUGGLE_RISE = 0.2  # m the ball must rise after a foot touch to count as a juggle


class SoccerJuggleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        frame_skip=10,
        max_steps=1000,
        drop_height=0.25,
        max_xy_dist=0.6,
        target_apex=1.05,
        width=640,
        height=480,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.drop_height = drop_height
        self.max_xy_dist = max_xy_dist
        self.target_apex = target_apex
        self.render_mode = render_mode
        self._width, self._height = width, height
        self._renderer = None
        self._viewer = None

        m = self.model
        self._leg_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in LEG_JOINTS])
        self._leg_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS])
        self._ball_qpos = m.jnt_qposadr[m.joint("ball").id]
        self._ball_qvel = m.jnt_dofadr[m.joint("ball").id]
        self._ball_geom = m.geom("ball").id
        self._foot_geom = m.geom("foot").id
        self._robot_geoms = {m.geom(g).id for g in ["foot", "shin", "thigh"]}
        self._floor_geom = m.geom("floor").id
        self._foot_site = m.site("foot_top").id

        self._home = np.radians(HOME_DEG)
        self._scale = np.radians(ACTION_SCALE_DEG)
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self._ctrl_lo, self._ctrl_hi = lo, hi

        # foot position in the home pose: the "juggling spot"
        self._set_leg(self._home)
        mujoco.mj_forward(m, self.data)
        self._home_foot = self.data.site_xpos[self._foot_site].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    def _set_leg(self, q):
        self.data.qpos[self._leg_qpos] = q
        self.data.ctrl[:] = q

    def _foot_state(self):
        pos = self.data.site_xpos[self._foot_site].copy()
        normal = self.data.site_xmat[self._foot_site].reshape(3, 3)[:, 2].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self._foot_site, vel, 0
        )
        return pos, normal, vel[3:]

    def _ball_state(self):
        p = self.data.qpos[self._ball_qpos : self._ball_qpos + 3].copy()
        v = self.data.qvel[self._ball_qvel : self._ball_qvel + 3].copy()
        return p, v

    def _contacts(self):
        """Return (foot_touch, leg_touch, floor_touch) for the ball this substep."""
        foot = leg = floor = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._ball_geom not in pair:
                continue
            other = (pair - {self._ball_geom}).pop()
            foot |= other == self._foot_geom
            leg |= other in self._robot_geoms and other != self._foot_geom
            floor |= other == self._floor_geom
        return foot, leg, floor

    def _obs(self):
        foot_p, foot_n, foot_v = self._foot_state()
        ball_p, ball_v = self._ball_state()
        return np.concatenate(
            [
                self.data.qpos[self._leg_qpos],
                self.data.qvel[self._leg_qvel] * 0.1,
                ball_p - foot_p,
                ball_v * 0.2,
                foot_n,
                foot_p - self._home_foot,
                foot_v * 0.2,
            ]
        )

    # --------------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        self._set_leg(self._home + rng.uniform(-0.05, 0.05, size=5))
        # drop the ball from above the juggling spot with a little noise
        ball = self._home_foot + np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.5, 0.8)]
        )
        self.data.qpos[self._ball_qpos : self._ball_qpos + 3] = ball
        self.data.qpos[self._ball_qpos + 3 : self._ball_qpos + 7] = [1, 0, 0, 0]
        self.data.qvel[self._ball_qvel : self._ball_qvel + 3] = [
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.5, 0.0),
        ]
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._touches = 0
        self._was_touching = False
        self._touch_z = None  # ball height at the last foot touch; pays out at the next apex
        self._prev_action = np.zeros(5)
        return self._obs(), {"touches": 0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = np.clip(self._home + action * self._scale, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = target

        new_touch = floor_hit = leg_hit = False
        contact_substeps = 0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            foot, leg, floor = self._contacts()
            if foot and not self._was_touching:
                new_touch = True
            self._was_touching = foot
            contact_substeps += foot or leg
            leg_hit |= leg
            floor_hit |= floor
        self._steps += 1

        ball_p, ball_v = self._ball_state()
        foot_p, foot_n, _ = self._foot_state()
        xy_err = np.linalg.norm(ball_p[:2] - self._home_foot[:2])

        # ---- reward ----
        reward = 0.0
        if contact_substeps == 0:
            # airborne: alive bonus + more the higher the ball is above the foot
            gap = ball_p[2] - foot_p[2] - BALL_RADIUS
            reward += 0.5 + 0.5 * np.clip(gap / 0.4, 0.0, 1.0)
        reward -= 0.05 * contact_substeps  # don't cradle / roll the ball on the leg
        reward -= 0.5 * leg_hit  # juggle with the foot, not the shin or thigh
        reward -= 1.0 * xy_err  # keep ball over the juggling spot
        # dense guidance: get foot under the ball while it is coming down
        if ball_v[2] < 0:
            reward -= 0.5 * np.linalg.norm(ball_p[:2] - foot_p[:2])
        if new_touch:
            self._touch_z = ball_p[2]
        # juggle bonus, paid at the apex after a foot touch if the ball really rose
        if self._touch_z is not None and ball_v[2] <= 0 and contact_substeps == 0:
            if ball_p[2] - self._touch_z > MIN_JUGGLE_RISE:
                self._touches += 1
                apex_err = ball_p[2] - self.target_apex
                reward += 2.0 + 4.0 * np.exp(-((apex_err / 0.2) ** 2))
            self._touch_z = None
        reward -= 0.05 * float(np.sum((action - self._prev_action) ** 2))  # smoothness
        self._prev_action = action

        dropped = ball_p[2] < self.drop_height or floor_hit
        lost = xy_err > self.max_xy_dist
        terminated = bool(dropped or lost)
        if terminated:
            reward -= 10.0
        truncated = self._steps >= self.max_steps

        info = {"touches": self._touches, "ball_height": ball_p[2], "xy_err": xy_err}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, self._height, self._width)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [0.25, -0.1, 0.6]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 2.6, 125, -12
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer  # on macOS, run the script with `mjpython`

            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


gym.register(id="SoccerJuggle-v0", entry_point="juggling.env:SoccerJuggleEnv")
