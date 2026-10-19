# CLAUDE.md — robotJuggling handoff

AA228 final project: reinforcement learning (Stable-Baselines3 PPO) that teaches a MuJoCo robot to juggle a soccer ball (keepie-uppie). This file records the project's state as of 2026-10-18, written for a Claude session picking the work up on another machine.

## Setup

```bash
conda env create -f environment.yml      # creates env "robotjuggling" (py3.11, mujoco, gymnasium, sb3, torch)
conda activate robotjuggling
```

Gotchas:
- `conda run` does **not** forward stdin (heredocs to `python -` silently do nothing), and it buffers stdout. For long runs use `conda run --no-capture-output -n robotjuggling python -u train.py ... > run.log 2>&1`.
- Speed: ~6–7k env steps/s with 8 `SubprocVecEnv` workers on an M4 Mac. The sim is CPU-bound, so a GPU does not help unless the env is ported to MJX (see Next steps).
- `runs/`, `*.mp4`, `*.log` are git-ignored. **Trained models and videos are not in git**: they exist only on the original Mac. Retrain to reproduce them.
- macOS live viewer needs `mjpython`. Off-screen `--record x.mp4` works everywhere (on headless Linux you may need `MUJOCO_GL=egl` or `osmesa`).

## Code map

| File | Purpose |
|---|---|
| `juggling/assets/juggler.xml` | Fixed torso, one 5-DOF kicking leg (hip pitch/roll, knee, ankle pitch/roll), size-5 ball (r=0.11 m, 0.43 kg, bounces back to ~0.9 of drop height) |
| `juggling/assets/juggler_2leg.xml` | Fixed torso, both legs actuated (10 DOF) |
| `juggling/assets/juggler_walk.xml` | Free-standing **planar** robot: torso slides x/z and pitches; hip/knee/ankle per leg; ball moves in the same x-z plane. Legs don't collide with each other (contype/conaffinity bits) |
| `juggling/env.py` | `SoccerJuggleEnv(legs=1\|2)`, the "static" task |
| `juggling/walk_env.py` | `SoccerJuggleWalkEnv(...)`, the "walk" task (balance + step to keep juggling) |
| `juggling/__init__.py` | `TASKS = {"static", "walk"}`, `make_env(task, **kwargs)` |
| `train.py` | PPO/SAC. `--task static\|walk`, `--legs 1\|2`, `--alternate`. Writes `runs/<name>/env.json`, checkpoints every 100k steps (+ VecNormalize), `best_model.zip` + `best_vecnormalize.pkl` |
| `enjoy.py` | Play/record a run (`--step N` for a checkpoint). Rebuilds the env from `env.json` (falls back to obs size for old runs) |
| `progress_video.py` | One labelled episode per checkpoint, stitched into one mp4 |

Shared reward ideas (both tasks): an alive bonus only while the ball is airborne, growing with height above the nearest foot; a "juggle" = a foot touch followed by ≥0.2 m rise, paying +2 plus up to +4 near `target_apex` (1.05 m); penalties for cradling (contact substeps), shin/thigh touches, ball drifting from the spot, and jerky actions; −10 on termination. PPO uses `log_std_init=-1` (the default std of 1 stalled learning completely).

## Results so far

| Run | Task | Outcome |
|---|---|---|
| `ppo_v3` | static, 1 leg, 4M steps | **Works**: full 20 s, 30 juggles/episode |
| `ppo_2leg_v1` | static, 2 legs, 8M | Failed: ~2 s, 1 juggle, right foot only. Reward not tuned for 10 DOF |
| `ppo_walk_v1` | walk, 10M | **Reward hack**: kicks the ball sky-high and sprints under it (up to 10 m). Best at 2.5M steps: 8–12 s, 2–4 kicks |
| `ppo_walk_v2` | walk + height cap, 10M | **Works**: full 20 s, 32 juggles/episode, balancing on the left leg and kicking with the right. Barely moves (~0.3 m) once trained |
| `ppo_walk_alt_v1` | walk `--alternate`, stopped at ~5M of 15M | **Broken, see Open problem** |

Fixes found along the way:
1. Ball initially landed on the knee: the home pose was moved so the foot sticks out in front.
2. The first reward (+1 alive) let the robot cradle the ball on its shin. It now pays only while the ball is airborne.
3. Exploration noise too large → `log_std_init=-1`.
4. `enjoy.py` paired `best_model` with end-of-training normalization stats → now uses `best_vecnormalize.pkl`.
5. The walk task's sky-kick hack → the airborne bonus stops above `max_bonus_height` = 1.5 m, plus `overshoot_penalty` = 5 per m above `target_apex` + 0.3.

## Open problem: alternating feet (`foot_command=True`)

Not yet committed before this handoff; it's included in the same commit as this file. In `walk_env.py`:
- The obs gains a one-hot "kick with this foot next" (35 dims instead of 33).
- Episodes start on a random stance leg.
- Actions are offsets from a symmetric pose (`SYM_HOME_DEG`), halfway between `RAISED_DEG` and `STANCE_DEG`.
- Only the commanded foot's juggles are rewarded; a wrong-foot juggle costs −3, and `max_wrong` = 3 of them end the episode.
- The command flips after `switch_every` correct juggles. `train.py --alternate` runs a curriculum (4 → 2 → 1 at 25%/50% of training); the eval env always uses 1.

Symptom: eval return stuck at ~100. At the 5M checkpoint the policy makes **zero** correct juggles. In every episode it kicks with the foot *opposite* the command, from the very first kick: it swaps stance at the start (e.g. starts standing on R with L raised and commanded, then ends up standing on L and kicking with R). Diagnostic output:

```
seed 0: start cmd L -> wrong-foot juggles by R at steps 36, 74 (grounded R/L = [False, True])
seed 1: start cmd R -> wrong-foot juggles by L at steps 43, 85
seed 2: start cmd L -> wrong-foot juggles by R at steps 39, 83, 120
```

The opposite foot every time looks like a bug rather than merely ignoring the command. Things to check first:
- Index consistency. The foot index is 0 = right and 1 = left in `_foot_geoms`, `_foot_sites`, `_grounded` and `_command`. `LEG_JOINTS` lists r_ first, and `reset()` builds `pose` as `[RAISED, STANCE]` when kick == 0.
- Whether "wrong-foot" juggles are actually the stance foot touching or bumping the ball while the robot swaps legs.
- Whether the symmetric home pose (action 0 = both legs half-raised) makes the robot fall unless it immediately picks one stance, so a fixed preferred stance wins out regardless of the command.
- Does a juggle by the commanded foot ever happen during training? Log the ratio.

If it's not a bug: consider a higher `target_apex` to give time for the weight shift, more training, or a warm start.

## Next steps (agreed with the user)

1. Fix and train alternating feet in the planar walk task (CPU is fine, ~30–60 min).
2. Full 3D robot that can fall sideways: freejoint torso, add hip-roll and ankle-roll joints (10 actuators), 3D ball. Use a lateral-support curriculum (a spring holding the torso upright side to side, fading out over training). Expect 50–100M steps: hours on CPU. The plan is to port the env to **MJX + Brax PPO and train on an NVIDIA GPU** (Colab, or whatever GPU the remote machine has) rather than SB3 on CPU.
3. Combine 3D with alternation.

## How the user likes to work

- Commit **and push** to `main` once each training run has finished (not mid-run).
- For every training run, make **progress videos** from the checkpoints (`progress_video.py`) and check the robot's behaviour from frames. Don't trust the reward alone: this project has already had two reward hacks.
- Report honestly when a run fails or is hacking the reward, and suggest the fix.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
