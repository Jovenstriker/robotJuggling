"""Watch or record a trained policy.

    python enjoy.py runs/ppo_s0 --record juggle.mp4   # offscreen video
    mjpython enjoy.py runs/ppo_s0                     # live viewer (macOS needs mjpython)
    python enjoy.py --random --legs 2 --record random.mp4  # random-action baseline
    python enjoy.py runs/ppo_s0 --step 1000000 --record early.mp4  # a training checkpoint
"""

import argparse
import pickle
import time
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO, SAC

from juggling import SoccerJuggleEnv


def load(run_dir: Path, step=None):
    """Return (policy, legs); the leg count is inferred from the saved obs size."""
    if step is not None:
        model_path = run_dir / "checkpoints" / f"rl_model_{step}_steps.zip"
        norm_path = run_dir / "checkpoints" / f"rl_model_vecnormalize_{step}_steps.pkl"
    else:
        model_path = run_dir / "best_model.zip"
        if not model_path.exists():
            model_path = run_dir / "final_model.zip"
        norm_path = run_dir / "vecnormalize.pkl"
    algo = SAC if "sac" in run_dir.name else PPO
    model = algo.load(model_path, device="cpu")
    with open(norm_path, "rb") as f:
        vn = pickle.load(f)
    obs_rms, clip = vn.obs_rms, vn.clip_obs

    def policy(obs):
        o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + vn.epsilon), -clip, clip)
        return model.predict(o, deterministic=True)[0]

    legs = (obs_rms.mean.shape[0] - 3) // 22
    return policy, legs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--random", action="store_true")
    p.add_argument("--step", type=int, default=None, help="load checkpoint at this timestep")
    p.add_argument("--record", type=str, default=None, help="output .mp4 path")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--legs", type=int, choices=[1, 2], default=1, help="only used with --random")
    args = p.parse_args()

    legs = args.legs
    if not args.random:
        policy, legs = load(args.run_dir, args.step)
    env = SoccerJuggleEnv(render_mode="rgb_array" if args.record else "human", legs=legs)
    if args.random:
        policy = lambda o: env.action_space.sample()

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
        print(f"episode {ep}: return={ret:.1f} touches={info['touches']} per-foot={info['foot_touches']} steps={env._steps}")
    if args.record:
        imageio.mimsave(args.record, frames, fps=env.metadata["render_fps"])
        print(f"wrote {args.record}")
    env.close()


if __name__ == "__main__":
    main()
