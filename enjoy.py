"""Watch or record a trained policy.

    python enjoy.py runs/ppo_s0 --record juggle.mp4   # offscreen video
    mjpython enjoy.py runs/ppo_s0                     # live viewer (macOS needs mjpython)
    python enjoy.py --random --record random.mp4      # random-action baseline
"""

import argparse
import pickle
import time
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO, SAC

from juggling import SoccerJuggleEnv


def load(run_dir: Path):
    model_path = run_dir / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    algo = SAC if "sac" in run_dir.name else PPO
    model = algo.load(model_path, device="cpu")
    with open(run_dir / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    obs_rms, clip = vn.obs_rms, vn.clip_obs

    def policy(obs):
        o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + vn.epsilon), -clip, clip)
        return model.predict(o, deterministic=True)[0]

    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--random", action="store_true")
    p.add_argument("--record", type=str, default=None, help="output .mp4 path")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    env = SoccerJuggleEnv(render_mode="rgb_array" if args.record else "human")
    policy = (lambda o: env.action_space.sample()) if args.random else load(args.run_dir)

    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            done = term or trunc
            if args.record:
                frames.append(env.render())
            else:
                time.sleep(env.frame_skip * env.model.opt.timestep)
        print(f"episode {ep}: return={ret:.1f} touches={info['touches']} steps={env._steps}")
    if args.record:
        imageio.mimsave(args.record, frames, fps=env.metadata["render_fps"])
        print(f"wrote {args.record}")
    env.close()


if __name__ == "__main__":
    main()
