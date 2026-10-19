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
| `juggling/assets/juggler_2leg.xml` | Same, with both legs actuated (10 DOF; red = right, blue = left boot) |
| `juggling/env.py` | Gymnasium env `SoccerJuggleEnv(legs=1|2)` (`SoccerJuggle-v0`, `SoccerJuggle2Leg-v0`) |
| `train.py` | PPO / SAC training with Stable-Baselines3 |
| `enjoy.py` | Watch a policy (or a training checkpoint) live or record an mp4 |
| `progress_video.py` | Stitch labelled episodes from several checkpoints into one video |

## Env details

- **Legs**: `legs=1` juggles on one spot with the right foot. `legs=2` treats the line between the two feet as the juggling spot, so the ball can travel sideways (y); consecutive juggles with different feet earn `alternate_bonus` (default +1).
- **Action** (5 per leg): joint position targets as offsets from a home pose, in [-1, 1]. Control at 50 Hz (10 physics substeps of 2 ms).
- **Observation** (22 per leg + 3): leg joint pos/vel, ball position relative to each foot, ball velocity, foot normals, foot position offsets, foot velocities.
- **Reward** per step: +0.5–1.0 while the ball is airborne (more the higher it is above the nearest foot), −0.05 per substep the ball touches the robot (no cradling), −0.5 for shin/thigh contact, −distance of ball from the juggling spot, −foot/ball horizontal misalignment while the ball falls. A juggle (foot touch followed by ≥0.2 m rise) pays +2 plus up to +4 for an apex near `target_apex` (1.05 m). Small action-smoothness penalty, −10 on drop.
- **Termination**: ball below 0.25 m or more than 0.6 m from the spot horizontally. Truncation at 1000 steps (20 s).

## Usage

```bash
python train.py --algo ppo --steps 5000000      # ~8 parallel envs
python train.py --algo sac --steps 1000000
python train.py --legs 2 --steps 8000000        # both legs
tensorboard --logdir runs

python enjoy.py runs/ppo_s0 --record juggle.mp4  # offscreen video
mjpython enjoy.py runs/ppo_s0                    # live viewer (macOS needs mjpython)
python enjoy.py runs/ppo_s0 --step 1000000 --record early.mp4   # a checkpoint
python enjoy.py --random --legs 2 --record random.mp4            # baseline
python progress_video.py runs/ppo_s0 --steps 100000 1000000 3000000 --out videos/progress.mp4
```
