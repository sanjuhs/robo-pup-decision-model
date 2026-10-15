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
    return { from: first.pose, to: second.pose, alpha };
  }
}

export function smoothingAlpha(dtSeconds, timeConstant) {
  return 1 - Math.exp(-Math.max(0, dtSeconds) / timeConstant);
}
