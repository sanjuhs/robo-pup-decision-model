import { test, expect } from '@playwright/test';
import { PoseBuffer, smoothingAlpha, updateOrbit } from '../src/motion.js';
import { Vector3 } from 'three';

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

test('paused render time stays bit-for-bit constant across fractional intervals', () => {
  const time = 4.599999999999924;
  const buffer = new PoseBuffer();
  buffer.push({time}, 0);
  buffer.push({time}, 40);
  for (let now = 60; now <= 100; now += .37) expect(buffer.sample(now).time).toBe(time);
});

test('idle orbit target is exact while real user/camera updates still move it', () => {
  const target=new Vector3(.668574373954221,-.052514854924089194,.1999982502624865);
  const previous=new Vector3();
  const initial=target.clone();
  const controls={target,update:()=>{target.x+=1e-16;return false;}};
  for(let i=0;i<100;i++)updateOrbit(controls,true,previous);
  expect(target.toArray()).toEqual(initial.toArray());
  controls.update=()=>{target.x+=.1;return true;};
  updateOrbit(controls,true,previous);
  expect(target.x).toBe(initial.x+.1);
  controls.update=()=>{target.y+=.1;return false;};
  updateOrbit(controls,false,previous);
  expect(target.y).toBe(initial.y+.1);
});
