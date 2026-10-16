"""RoboPup Arena: an honest, inspectable robot pet research prototype."""

__version__ = "0.1.0"
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
