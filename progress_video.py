"""Stitch one labelled episode per training checkpoint into a single video.

    python progress_video.py runs/ppo_v3 --steps 100000 1000000 2000000 3000000 --out videos/progress.mp4
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from enjoy import load
from juggling import SoccerJuggleEnv


def label(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    draw.text((14, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0, help="same ball drop for every checkpoint")
    p.add_argument("--out", default="videos/progress.mp4")
    args = p.parse_args()

    env = None
    frames = []
    for step in args.steps:
        policy, legs = load(args.run_dir, step)
        if env is None:
            env = SoccerJuggleEnv(render_mode="rgb_array", legs=legs)
        obs, _ = env.reset(seed=args.seed)
        done, info = False, {"touches": 0}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
            t = env._steps / env.metadata["render_fps"]
            frames.append(label(env.render(),
                                f"{step / 1e6:.1f}M steps   juggles: {info['touches']}   t = {t:4.1f}s"))
        frames += [frames[-1]] * 25  # hold the last frame for 0.5 s
        print(f"{step}: juggles={info['touches']} steps={env._steps}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=env.metadata["render_fps"])
    print(f"wrote {args.out} ({len(frames) / env.metadata['render_fps']:.0f} s)")
    env.close()


if __name__ == "__main__":
    main()
