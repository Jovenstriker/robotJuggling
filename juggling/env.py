"""Gymnasium environment: a fixed-base robot juggling a soccer ball in MuJoCo.

`legs=1`: only the right leg is actuated (juggler.xml); the ball is juggled on a spot.
`legs=2`: both legs are actuated (juggler_2leg.xml); the ball may travel sideways
          between the feet, and alternating feet earns `alternate_bonus`.

Action  (5 * legs,) in [-1, 1]: joint position targets, offset from a "home" pose
              (hip pitch, hip roll, knee, ankle pitch, ankle roll) per leg.
Obs     (22 * legs + 3,): leg qpos, leg qvel, ball pos rel. each foot, ball vel,
               foot normals, foot pos rel. home, foot linear vels.
Reward favors an airborne ball, real bounces off a foot (counted as "touches"
in info) with an apex near `target_apex`, and penalizes cradling the ball.
Episode ends when the ball drops below `drop_height`, drifts too far, or after
`max_steps` control steps (20 s at 50 Hz by default).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSETS = Path(__file__).parent / "assets"
XML_PATHS = {1: ASSETS / "juggler.xml", 2: ASSETS / "juggler_2leg.xml"}
LEG_PREFIXES = {1: [""], 2: ["r_", "l_"]}
LEG_JOINTS = ["hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]
HOME_DEG = np.array([-75.0, 0.0, 35.0, 40.0, 0.0])
# how far (deg) the policy may move each joint target away from home
ACTION_SCALE_DEG = np.array([45.0, 25.0, 60.0, 40.0, 25.0])
BALL_RADIUS = 0.11
MIN_JUGGLE_RISE = 0.2  # m the ball must rise after a foot touch to count as a juggle


def _dist_to_segment(p, a, b):
    """Distance from point p to segment ab (a point when a == b), all in the xy plane."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else np.clip((p - a) @ ab / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class SoccerJuggleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        legs=1,
        frame_skip=10,
        max_steps=1000,
        drop_height=0.25,
        max_xy_dist=0.6,
        target_apex=1.05,
        alternate_bonus=1.0,
        width=640,
        height=480,
    ):
        if legs not in XML_PATHS:
            raise ValueError(f"legs must be 1 or 2, got {legs}")
        self.legs = legs
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATHS[legs]))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.drop_height = drop_height
        self.max_xy_dist = max_xy_dist
        self.target_apex = target_apex
        self.alternate_bonus = alternate_bonus
        self.render_mode = render_mode
        self._width, self._height = width, height
        self._renderer = None
        self._viewer = None

        m = self.model
        prefixes = LEG_PREFIXES[legs]
        joints = [p + j for p in prefixes for j in LEG_JOINTS]
        self._leg_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in joints])
        self._leg_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in joints])
        self._ball_qpos = m.jnt_qposadr[m.joint("ball").id]
        self._ball_qvel = m.jnt_dofadr[m.joint("ball").id]
        self._ball_geom = m.geom("ball").id
        self._foot_geoms = [m.geom(p + "foot").id for p in prefixes]
        self._robot_geoms = {m.geom(p + g).id for p in prefixes for g in ["foot", "shin", "thigh"]}
        self._floor_geom = m.geom("floor").id
        self._foot_sites = [m.site(p + "foot_top").id for p in prefixes]

        self._home = np.tile(np.radians(HOME_DEG), legs)
        self._scale = np.tile(np.radians(ACTION_SCALE_DEG), legs)
        self._ctrl_lo, self._ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

        # foot positions in the home pose; the juggling spot is the segment between them
        self._set_leg(self._home)
        mujoco.mj_forward(m, self.data)
        self._home_feet = np.array([self.data.site_xpos[s].copy() for s in self._foot_sites])

        n_act = 5 * legs
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(22 * legs + 3,), dtype=np.float64
        )

    # ------------------------------------------------------------------ helpers
    def _set_leg(self, q):
        self.data.qpos[self._leg_qpos] = q
        self.data.ctrl[:] = q

    def _feet_state(self):
        """Arrays (legs, 3) of foot-top positions, normals and linear velocities."""
        pos, normal, vel = [], [], []
        v6 = np.zeros(6)
        for s in self._foot_sites:
            pos.append(self.data.site_xpos[s].copy())
            normal.append(self.data.site_xmat[s].reshape(3, 3)[:, 2].copy())
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, s, v6, 0)
            vel.append(v6[3:].copy())
        return np.array(pos), np.array(normal), np.array(vel)

    def _ball_state(self):
        p = self.data.qpos[self._ball_qpos : self._ball_qpos + 3].copy()
        v = self.data.qvel[self._ball_qvel : self._ball_qvel + 3].copy()
        return p, v

    def _spot_dist(self, ball_xy):
        return _dist_to_segment(ball_xy, self._home_feet[0, :2], self._home_feet[-1, :2])

    def _contacts(self):
        """Return (feet_touch per foot, leg_touch, floor_touch) for the ball this substep."""
        feet = [False] * self.legs
        leg = floor = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._ball_geom not in pair:
                continue
            other = (pair - {self._ball_geom}).pop()
            if other in self._foot_geoms:
                feet[self._foot_geoms.index(other)] = True
            elif other in self._robot_geoms:
                leg = True
            floor |= other == self._floor_geom
        return feet, leg, floor

    def _obs(self):
        feet_p, feet_n, feet_v = self._feet_state()
        ball_p, ball_v = self._ball_state()
        return np.concatenate(
            [
                self.data.qpos[self._leg_qpos],
                self.data.qvel[self._leg_qvel] * 0.1,
                (ball_p - feet_p).ravel(),
                ball_v * 0.2,
                feet_n.ravel(),
                (feet_p - self._home_feet).ravel(),
                feet_v.ravel() * 0.2,
            ]
        )

    # --------------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        self._set_leg(self._home + rng.uniform(-0.05, 0.05, size=self._home.size))
        # drop the ball from above a random point of the juggling spot
        t = rng.uniform(0.0, 1.0)
        spot = (1 - t) * self._home_feet[0] + t * self._home_feet[-1]
        ball = spot + np.array(
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
        self._foot_touches = [0] * self.legs
        self._was_touching = [False] * self.legs
        self._touch_z = None  # ball height at the last foot touch; pays out at the next apex
        self._touch_foot = None  # which foot made that touch
        self._last_juggle_foot = None
        self._prev_action = np.zeros(self.action_space.shape)
        return self._obs(), {"touches": 0, "foot_touches": list(self._foot_touches)}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = np.clip(self._home + action * self._scale, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = target

        new_touch_foot = None
        floor_hit = leg_hit = False
        contact_substeps = 0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            feet, leg, floor = self._contacts()
            for k, touching in enumerate(feet):
                if touching and not self._was_touching[k]:
                    new_touch_foot = k
            self._was_touching = feet
            contact_substeps += any(feet) or leg
            leg_hit |= leg
            floor_hit |= floor
        self._steps += 1

        ball_p, ball_v = self._ball_state()
        feet_p, _, _ = self._feet_state()
        xy_err = self._spot_dist(ball_p[:2])
        feet_xy_err = np.linalg.norm(ball_p[:2] - feet_p[:, :2], axis=1)
        nearest = int(np.argmin(feet_xy_err))

        # ---- reward ----
        reward = 0.0
        if contact_substeps == 0:
            # airborne: alive bonus + more the higher the ball is above the nearest foot
            gap = ball_p[2] - feet_p[nearest, 2] - BALL_RADIUS
            reward += 0.5 + 0.5 * np.clip(gap / 0.4, 0.0, 1.0)
        reward -= 0.05 * contact_substeps  # don't cradle / roll the ball on the leg
        reward -= 0.5 * leg_hit  # juggle with the feet, not the shins or thighs
        reward -= 1.0 * xy_err  # keep ball over the juggling spot
        # dense guidance: get a foot under the ball while it is coming down
        if ball_v[2] < 0:
            reward -= 0.5 * feet_xy_err[nearest]
        if new_touch_foot is not None:
            self._touch_z = ball_p[2]
            self._touch_foot = new_touch_foot
        # juggle bonus, paid at the apex after a foot touch if the ball really rose
        if self._touch_z is not None and ball_v[2] <= 0 and contact_substeps == 0:
            if ball_p[2] - self._touch_z > MIN_JUGGLE_RISE:
                self._touches += 1
                self._foot_touches[self._touch_foot] += 1
                apex_err = ball_p[2] - self.target_apex
                reward += 2.0 + 4.0 * np.exp(-((apex_err / 0.2) ** 2))
                if self._last_juggle_foot is not None and self._touch_foot != self._last_juggle_foot:
                    reward += self.alternate_bonus
                self._last_juggle_foot = self._touch_foot
            self._touch_z = None
        reward -= 0.05 * float(np.sum((action - self._prev_action) ** 2))  # smoothness
        self._prev_action = action

        dropped = ball_p[2] < self.drop_height or floor_hit
        lost = xy_err > self.max_xy_dist
        terminated = bool(dropped or lost)
        if terminated:
            reward -= 10.0
        truncated = self._steps >= self.max_steps

        info = {
            "touches": self._touches,
            "foot_touches": list(self._foot_touches),
            "ball_height": ball_p[2],
            "xy_err": xy_err,
        }
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, self._height, self._width)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [0.25, self._home_feet[:, 1].mean(), 0.6]
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
gym.register(id="SoccerJuggle2Leg-v0", entry_point="juggling.env:SoccerJuggleEnv", kwargs={"legs": 2})
