// Render one shared interval for every body. Never extrapolate a physical pose.
export class PoseBuffer {
  constructor(delayMs = 60) {
    this.delayMs = delayMs;
    this.frames = [];
  }

  push(pose, receivedAt) {
    const previous = this.frames.at(-1);
    // A simulation reset must snap to the new world, not sweep through the yard.
    if (previous && pose.time < previous.pose.time) this.frames.length = 0;
    this.frames.push({ pose, receivedAt });
    if (this.frames.length > 12) this.frames.shift();
  }

  sample(now) {
    if (!this.frames.length) return null;
    const at = now - this.delayMs;
    while (this.frames.length > 2 && this.frames[1].receivedAt <= at) this.frames.shift();
    const first = this.frames[0];
    const second = this.frames[1] ?? first;
    const span = second.receivedAt - first.receivedAt;
    const alpha = span > 0 ? Math.max(0, Math.min(1, (at - first.receivedAt) / span)) : 0;
    // Unlike a*(1-alpha)+b*alpha, this stays bit-for-bit constant when paused.
    const time = first.pose.time + (second.pose.time - first.pose.time) * alpha;
    return { from: first.pose, to: second.pose, alpha, time };
  }
}

export function smoothingAlpha(dtSeconds, timeConstant) {
  return 1 - Math.exp(-Math.max(0, dtSeconds) / timeConstant);
}

export function updateOrbit(controls, preserveIdleTarget, previousTarget) {
  previousTarget.copy(controls.target);
  const changed = controls.update();
  // OrbitControls clamps target radius via normalize/multiply even with no
  // radius limits. Avoid accumulating round-off while the camera is idle.
  if (preserveIdleTarget && !changed) controls.target.copy(previousTarget);
}
