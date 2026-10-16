#!/usr/bin/env python3
"""Measure independent live worlds against an already running local server."""

import argparse
import asyncio
import json
from pathlib import Path
import time

from websockets.asyncio.client import connect


async def participant(index, frames, activity):
    async with connect("ws://127.0.0.1:8000/ws") as ws:
        scene = json.loads(await ws.recv())
        assert scene["type"] == "scene" and len(scene["geoms"]) > 12
        target = [index * 0.2, -index * 0.1]
        await ws.send(json.dumps({"mode": "stop", "target": target}))
        if activity == "fetch":
            await ws.send(json.dumps({"text": "fetch the ball"}))
        while True:
            first = json.loads(await ws.recv())
            if first.get("target") == target and (
                activity == "stop" or (first.get("planner") or {}).get("plan")
            ):
                if activity == "fetch":
                    assert first["brain"] == "neural" and first["planner"]["intent"] == "fetch"
                break
        start = time.perf_counter()
        for _ in range(frames):
            last = json.loads(await ws.recv())
            if last.get("type") != "state":
                continue
            assert last["target"] == target and not last["needs_reset"]
            if activity == "fetch":
                assert last["planner"]["plan"] and last["planner"]["camera_png"]
        elapsed = time.perf_counter() - start
        simulated = last["time"] - first["time"]
        return {
            "participant": index,
            "frames": frames,
            "wall_seconds": elapsed,
            "simulated_seconds": simulated,
            "real_time_ratio": simulated / elapsed,
        }


async def run(count, frames, activity):
    return await asyncio.gather(*(participant(i, frames, activity) for i in range(count)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participants", type=int, choices=range(1, 13), default=10)
    p.add_argument("--frames", type=int, default=75)
    p.add_argument("--activity", choices=["stop", "fetch"], default="stop")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = {
        "protocol": "independent-" + args.activity + "-worlds-v2",
        "participants": args.participants,
        "sessions": asyncio.run(run(args.participants, args.frames, args.activity)),
        "note": "Short local WebSocket capacity probe, not a two-hour browser soak test",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("Keep previous evidence; choose a new output path")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "participants": args.participants,
                "min_real_time_ratio": min(s["real_time_ratio"] for s in result["sessions"]),
            }
        )
    )


if __name__ == "__main__":
    main()
