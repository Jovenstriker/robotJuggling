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
| `juggling/assets/juggler_walk.xml` | Free-standing planar robot (torso moves in x, z, pitch; hip/knee/ankle per leg); ball moves in the same plane |
| `juggling/walk_env.py` | Gymnasium env `SoccerJuggleWalkEnv` (`SoccerJuggleWalk-v0`): balance and step to keep juggling |
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

### Walking juggler (`--task walk`)

The torso is no longer fixed: it can move forward/back, up/down and pitch, and has to be balanced on the feet (it cannot fall sideways). It starts standing on the left leg with the right foot raised. The ball is launched with a random forward/backward speed (`ball_speed`, default 0.3 m/s), so the robot has to step or hop to stay under it. Walking is never rewarded directly; it only moves to keep juggling.

- **Action** (6): hip/knee/ankle targets per leg. **Observation** (33): torso height, pitch, velocities, joints, ball relative to torso and each foot, foot velocities and pitch, foot-on-ground flags.
- **Reward**: the juggling terms above, measured relative to the robot (ball should stay within 0.25 m of the spot in front of it), plus +0.2 per step for staying up and −0.5·|torso pitch|.
- **Termination**: ball dropped or more than 0.7 m beyond reach, or the robot falls (torso below 0.6 m or pitch > 0.8 rad).

## Usage

```bash
python train.py --algo ppo --steps 5000000      # ~8 parallel envs
python train.py --algo sac --steps 1000000
python train.py --legs 2 --steps 8000000        # both legs
python train.py --task walk --steps 10000000    # free-standing robot
tensorboard --logdir runs

python enjoy.py runs/ppo_s0 --record juggle.mp4  # offscreen video
mjpython enjoy.py runs/ppo_s0                    # live viewer (macOS needs mjpython)
python enjoy.py runs/ppo_s0 --step 1000000 --record early.mp4   # a checkpoint
python enjoy.py --random --legs 2 --record random.mp4            # baseline
python enjoy.py --random --task walk --record random_walk.mp4
python progress_video.py runs/ppo_s0 --steps 100000 1000000 3000000 --out videos/progress.mp4
```
