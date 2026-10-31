"""Generates speech with espeak-ng, trains the wake-word model, and tests it on unseen voices.

The wake phrase is "hey nova". Negatives include near misses like "hey nora". Voices are split, so
the test set contains only voices the model never heard in training.

usage: python -m wakeword.train --out wakeword/artifacts
"""
import argparse
import copy
import json
import pathlib
import random
import subprocess
import tempfile
import wave

import numpy as np
import torch
from torch import nn

from wakeword.model import CLIP_SAMPLES, SAMPLE_RATE, WakeWordNet, parameter_count

WAKE = ["hey nova", "hey, nova", "hey nova!"]
NEAR_MISS = ["hey nora", "hey over", "hey november", "say nova", "hey no", "nova", "hey", "a nova", "hay novel"]
OTHER = [
    "good morning", "turn on the lights", "what time is it", "open the door", "call the office",
    "the signal is weak", "check the antenna", "plasma physics", "thank you very much", "one two three",
    "how are you today", "see you tomorrow", "play some music", "stop", "yes", "no", "hello there",
]
BASES = ["en-us", "en-gb", "en-gb-scotland", "en-029", "en-gb-x-rp"]
VARIANTS = ["m1", "m2", "m3", "m4", "m5", "m7", "f1", "f2", "f3", "f4", "f5", "klatt", "klatt2"]
HELD_OUT = {"m4", "f3", "klatt2"}  # These voices appear only in validation and test.


def synthesize(text, voice, speed, pitch, folder):
    path = pathlib.Path(folder) / "clip.wav"
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", str(path), text],
        check=True, capture_output=True,
    )
    with wave.open(str(path)) as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    positions = np.arange(0, len(audio), rate / SAMPLE_RATE)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def fit_clip(audio, rng):
    """Places speech at a random position in a fixed-length clip."""
    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    audio = audio[:CLIP_SAMPLES]
    start = rng.randint(0, CLIP_SAMPLES - len(audio))
    clip[start:start + len(audio)] = audio
    return clip


def augment(clip, rng):
    """Random volume and background noise, because real microphones aren't clean."""
    noise_level = rng.uniform(0.002, 0.03)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, noise_level, len(clip)).astype(np.float32)
    return clip * rng.uniform(0.3, 1.0) + noise


def build_dataset(voices, repeats, rng):
    """Returns (clips, labels). Half of the negative examples are near misses."""
    clips, labels = [], []
    with tempfile.TemporaryDirectory() as folder:
        for voice in voices:
            for _ in range(repeats):
                speed, pitch = rng.randint(120, 200), rng.randint(25, 75)
                for text, label in (
                    [(rng.choice(WAKE), 1.0) for _ in range(4)]
                    + [(rng.choice(NEAR_MISS), 0.0) for _ in range(2)]
                    + [(rng.choice(OTHER), 0.0) for _ in range(2)]
                ):
                    audio = synthesize(text, voice, speed, pitch, folder)
                    clips.append(augment(fit_clip(audio, rng), rng))
                    labels.append(label)
    return torch.tensor(np.stack(clips)), torch.tensor(labels)


def voices_for(variants):
    return [f"{base}+{variant}" for base in BASES for variant in variants]


@torch.no_grad()
def scores(model, clips):
    model.eval()
    return torch.cat([torch.sigmoid(model(batch)) for batch in clips.split(128)])


def rates(probabilities, labels, threshold):
    """False accept rate (woke up wrongly) and false reject rate (ignored the wake phrase)."""
    predicted = probabilities >= threshold
    negatives, positives = labels == 0, labels == 1
    return {
        "false_accept": float(predicted[negatives].float().mean()),
        "false_reject": float((~predicted[positives]).float().mean()),
        "accuracy": float((predicted == positives).float().mean()),
    }


def pick_threshold(probabilities, labels):
    """A false wake-up annoys more than repeating yourself, so false accepts weigh twice as much."""
    candidates = torch.linspace(0.05, 0.95, 19)
    def cost(threshold):
        r = rates(probabilities, labels, threshold)
        return 2 * r["false_accept"] + r["false_reject"]
    return float(min(candidates, key=cost))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="wakeword/artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    held_out = sorted(HELD_OUT)
    train_x, train_y = build_dataset(voices_for([v for v in VARIANTS if v not in HELD_OUT]), args.repeats, rng)
    val_x, val_y = build_dataset(voices_for(held_out[:1]), args.repeats, rng)
    test_x, test_y = build_dataset(voices_for(held_out[1:]), args.repeats, rng)

    model = WakeWordNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    batches_per_epoch = (len(train_x) + 63) // 64
    # Ramping the learning rate down late in training stops scores swinging between epochs.
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=args.epochs, steps_per_epoch=batches_per_epoch,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_cost = None, float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss) * len(batch)
        # Keep the best epoch on voices the model doesn't train on, not simply the last one.
        val_scores = scores(model, val_x)
        val = rates(val_scores, val_y, pick_threshold(val_scores, val_y))
        cost = 2 * val["false_accept"] + val["false_reject"]
        if cost < best_cost:
            best_cost, best_state = cost, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: loss {total / len(train_x):.3f}, validation cost {cost:.3f}", flush=True)
    model.load_state_dict(best_state)

    # The threshold comes from validation voices, so the test score stays honest.
    threshold = pick_threshold(scores(model, val_x), val_y)
    test = rates(scores(model, test_x), test_y, threshold)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(out / "wakeword.pt"))
    metrics = {
        "threshold": threshold, "test": test, "parameters": parameter_count(model),
        "model_kb": round((out / "wakeword.pt").stat().st_size / 1024, 1),
        "train_clips": len(train_x), "test_clips": len(test_x), "test_voices": held_out[1:],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
