"""Train a policy to juggle a soccer ball.

    python train.py --algo ppo --steps 5_000_000
    python train.py --algo sac --steps 1_000_000
    tensorboard --logdir runs
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from juggling import SoccerJuggleEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="run name (default: <algo>_s<seed>)")
    args = p.parse_args()

    name = args.name or f"{args.algo}_s{args.seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    n_envs = args.n_envs if args.algo == "ppo" else max(1, args.n_envs // 2)
    env = make_vec_env(SoccerJuggleEnv, n_envs=n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = make_vec_env(SoccerJuggleEnv, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", env, n_steps=2048, batch_size=512, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
            tensorboard_log=str(out), seed=args.seed, verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy", env, learning_rate=3e-4, buffer_size=1_000_000,
            batch_size=256, gamma=0.99, tau=0.005, learning_starts=10_000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out), seed=args.seed, verbose=1,
        )

    class SyncNorm(EvalCallback):
        """Copy obs-normalization stats into the eval env before each eval."""
        def _on_step(self):
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_env.obs_rms = env.obs_rms
            return super()._on_step()

    callbacks = [
        CheckpointCallback(max(100_000 // n_envs, 1), str(out / "checkpoints"),
                           save_vecnormalize=True),
        SyncNorm(eval_env, best_model_save_path=str(out), log_path=str(out),
                 eval_freq=max(50_000 // n_envs, 1), n_eval_episodes=5, deterministic=True),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save(out / "final_model")
    env.save(str(out / "vecnormalize.pkl"))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
