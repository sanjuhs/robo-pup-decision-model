import { test, expect } from '@playwright/test';
import { PoseBuffer, smoothingAlpha } from '../src/motion.js';

test('pose playback is independent of render frequency and bounded by actual snapshots', () => {
  for (const hz of [30, 60, 144]) {
    const buffer = new PoseBuffer();
    for (const at of [0, 40, 80, 120]) buffer.push({time: at / 1000}, at);
    for (let at = 0; at < 150; at += 1000 / hz) buffer.sample(at);
    const {from, to, alpha} = buffer.sample(150);
    expect(from.time + (to.time - from.time) * alpha).toBeCloseTo(.09, 8);
    const stalled = buffer.sample(1000);
    expect(stalled.to.time).toBe(.12);
    expect(stalled.alpha).toBe(1);
  }
});

test('reset discards prior poses; pause holds the physical pose', () => {
  const buffer = new PoseBuffer();
  buffer.push({time: 10}, 0);
  buffer.push({time: 10.04}, 40);
  buffer.push({time: 0}, 80);
  expect(buffer.sample(90).from.time).toBe(0);
  buffer.push({time: 0}, 120);
  const sample = buffer.sample(170);
  expect(sample.from.time).toBe(0);
  expect(sample.to.time).toBe(0);
});

test('camera smoothing has the same response across refresh rates', () => {
  for (const hz of [30, 60, 144]) {
    let position = 0;
    for (let frame = 0; frame < hz; frame++) position += (1 - position) * smoothingAlpha(1 / hz, .32);
    expect(position).toBeCloseTo(1 - Math.exp(-1 / .32), 10);
  }
});
