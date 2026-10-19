# robotJuggling

Reinforcement learning for a robot leg that juggles a soccer ball ("keepie-uppie") in MuJoCo.

## Setup

```bash
conda env create -f environment.yml   # or: already created as `robotjuggling`
conda activate robotjuggling
```

## Layout

| Path | What |
|---|---|
| `juggling/assets/juggler.xml` | MuJoCo model: fixed torso, 5-DOF kicking leg (hip pitch/roll, knee, ankle pitch/roll), size-5 soccer ball (r=0.11 m, 0.43 kg) |
| `juggling/env.py` | Gymnasium env `SoccerJuggleEnv` (`SoccerJuggle-v0`) |
| `train.py` | PPO / SAC training with Stable-Baselines3 |
| `enjoy.py` | Watch a policy live or record an mp4 |

## Env details

- **Action** (5): joint position targets as offsets from a home pose, in [-1, 1]. Control at 50 Hz (10 physics substeps of 2 ms).
- **Observation** (25): leg joint pos/vel, ball position relative to foot, ball velocity, foot normal, foot position offset, foot velocity.
- **Reward** per step: +1 alive, −distance of ball from the juggling spot, −foot/ball horizontal misalignment while ball falls, +2 per upward touch, +up to 3 for an apex near `target_apex` (0.9 m), small action-smoothness penalty, −10 on drop.
- **Termination**: ball below 0.25 m or more than 0.6 m from the spot horizontally. Truncation at 1000 steps (20 s).

## Usage

```bash
python train.py --algo ppo --steps 5000000      # ~8 parallel envs
python train.py --algo sac --steps 1000000
tensorboard --logdir runs

python enjoy.py runs/ppo_s0 --record juggle.mp4  # offscreen video
mjpython enjoy.py runs/ppo_s0                    # live viewer (macOS needs mjpython)
python enjoy.py --random --record random.mp4     # baseline
```
