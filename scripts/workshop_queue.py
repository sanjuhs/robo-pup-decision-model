#!/usr/bin/env python3
"""Run participant policies sequentially; never provision or bill infrastructure."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participants", type=int, choices=range(5, 11), default=5)
    p.add_argument("--event", default="workshop-01")
    p.add_argument("--execute", action="store_true", help="Without this, print commands only")
    args = p.parse_args()
    for i in range(args.participants):
        command = [sys.executable, "-m", "robopup.train", "--participant", f"p{i+1:02}",
                   "--run", args.event, "--config", "configs/curious.json",
                   "--seed", str(100+i), "--max-minutes", "8"]
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, timeout=10*60)


if __name__ == "__main__":
    main()
