"""Watch or record a trained policy.

    python enjoy.py runs/ppo_s0 --record juggle.mp4   # offscreen video
    mjpython enjoy.py runs/ppo_s0                     # live viewer (macOS needs mjpython)
    python enjoy.py --random --legs 2 --record random.mp4  # random-action baseline
    python enjoy.py runs/ppo_s0 --step 1000000 --record early.mp4  # a training checkpoint
    python enjoy.py --random --task walk --record random_walk.mp4
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO, SAC

from juggling import TASKS, make_env

# fallback for runs saved before env.json existed: identify the env by obs size
_OBS_DIM_TO_ENV = {25: ("static", {"legs": 1}), 47: ("static", {"legs": 2}), 33: ("walk", {})}


def _best_norm_path(run_dir: Path) -> Path:
    """Normalization stats matching best_model.zip (not the end of training)."""
    if (run_dir / "best_vecnormalize.pkl").exists():
        return run_dir / "best_vecnormalize.pkl"
    # older runs: use the checkpoint saved closest after the best evaluation
    evals = np.load(run_dir / "evaluations.npz")
    best_step = int(evals["timesteps"][np.argmax(evals["results"].mean(axis=1))])
    ckpts = sorted(
        (int(p.stem.split("_")[-2]), p)
        for p in (run_dir / "checkpoints").glob("rl_model_vecnormalize_*_steps.pkl")
    )
    after = [p for s, p in ckpts if s >= best_step]
    return after[0] if after else run_dir / "vecnormalize.pkl"


def load(run_dir: Path, step=None):
    """Return (policy, (task, env_kwargs)) for a run directory."""
    if step is not None:
        model_path = run_dir / "checkpoints" / f"rl_model_{step}_steps.zip"
        norm_path = run_dir / "checkpoints" / f"rl_model_vecnormalize_{step}_steps.pkl"
    else:
        model_path = run_dir / "best_model.zip"
        norm_path = _best_norm_path(run_dir)
        if not model_path.exists():
            model_path, norm_path = run_dir / "final_model.zip", run_dir / "vecnormalize.pkl"
    algo = SAC if "sac" in run_dir.name else PPO
    model = algo.load(model_path, device="cpu")
    with open(norm_path, "rb") as f:
        vn = pickle.load(f)
    obs_rms, clip = vn.obs_rms, vn.clip_obs

    def policy(obs):
        o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + vn.epsilon), -clip, clip)
        return model.predict(o, deterministic=True)[0]

    cfg_path = run_dir / "env.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        env_cfg = (cfg["task"], cfg["kwargs"])
    else:
        env_cfg = _OBS_DIM_TO_ENV[obs_rms.mean.shape[0]]
    return policy, env_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--random", action="store_true")
    p.add_argument("--step", type=int, default=None, help="load checkpoint at this timestep")
    p.add_argument("--record", type=str, default=None, help="output .mp4 path")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--task", choices=list(TASKS), default="static", help="only used with --random")
    p.add_argument("--legs", type=int, choices=[1, 2], default=1, help="only used with --random")
    args = p.parse_args()

    if args.random:
        task, kwargs = args.task, ({"legs": args.legs} if args.task == "static" else {})
    else:
        policy, (task, kwargs) = load(args.run_dir, args.step)
    env = make_env(task, render_mode="rgb_array" if args.record else "human", **kwargs)
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
        extra = f" moved={info['distance']:+.2f}m" if "distance" in info else ""
        print(f"episode {ep}: return={ret:.1f} touches={info['touches']} "
              f"per-foot={info['foot_touches']} steps={env._steps}{extra}")
    if args.record:
        imageio.mimsave(args.record, frames, fps=env.metadata["render_fps"])
        print(f"wrote {args.record}")
    env.close()


if __name__ == "__main__":
    main()
