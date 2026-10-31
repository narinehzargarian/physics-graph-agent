"""Scores raw 16-bit mono 16 kHz audio with the exported model. Run with: uvicorn wakeword.service:app"""
import json
import os
import pathlib

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request

from wakeword.model import CLIP_SAMPLES

ARTIFACTS = pathlib.Path(os.environ.get("WAKEWORD_ARTIFACTS", "wakeword/artifacts"))
MAX_BYTES = CLIP_SAMPLES * 2

app = FastAPI(title="Wake Word")
loaded = {}


def model():
    if not loaded:
        loaded["model"] = torch.jit.load(str(ARTIFACTS / "wakeword.pt")).eval()
        loaded["threshold"] = json.loads((ARTIFACTS / "metrics.json").read_text())["threshold"]
    return loaded["model"], loaded["threshold"]


def to_clip(body):
    """Pads short audio and rejects anything too long or not made of 16-bit samples."""
    if not body or len(body) > MAX_BYTES or len(body) % 2:
        raise ValueError("expected up to 1.5 s of 16-bit mono audio")
    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    clip[:len(audio)] = audio
    return torch.from_numpy(clip).unsqueeze(0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(request: Request):
    try:
        clip = to_clip(await request.body())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    net, threshold = model()
    with torch.no_grad():
        score = float(torch.sigmoid(net(clip))[0])
    return {"score": round(score, 4), "detected": score >= threshold}
