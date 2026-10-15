import { test, expect } from '@playwright/test';
import fs from 'node:fs';

test('renders live geometry, behavior controls, pause, reset and keyboard', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width:1440,height:1050});
  await page.goto('/');
  await expect(page.getByText('Arena connected')).toBeVisible();
  await page.waitForFunction(()=>window.arenaTelemetry?.time>.3);
  expect(await page.evaluate(()=>window.arenaTelemetry.geometryCount)).toBeGreaterThan(12);
  await page.getByRole('button',{name:/Find a treat/}).click();
  await page.waitForFunction(()=>window.arenaTelemetry.behavior==='approach');
  await page.getByRole('button',{name:'Pause simulation'}).click();
  await page.waitForTimeout(150);
  const paused = await page.evaluate(()=>window.arenaTelemetry.time);
  await page.waitForTimeout(200);
  expect(await page.evaluate(()=>window.arenaTelemetry.time)).toBe(paused);
  await page.getByRole('button',{name:'Resume simulation'}).click();
  await page.getByRole('button',{name:/Reset posture/}).click();
  await page.keyboard.down('w');
  await page.waitForFunction(()=>window.arenaTelemetry.behavior==='manual');
  await page.keyboard.up('w');
  await page.keyboard.press('Space');
  await page.waitForFunction(()=>window.arenaTelemetry.behavior==='stop');
  await page.waitForTimeout(350);
  fs.mkdirSync('../artifacts/screens',{recursive:true});
  await page.screenshot({path:'../artifacts/screens/desktop.png',fullPage:true});
  expect(errors).toEqual([]);
});

test('shows real experiment results and can load a trusted checkpoint', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(()=>window.arenaTelemetry);
  await page.getByRole('button',{name:/Experiments/}).click();
  const runs = await (await page.request.get('/api/runs')).json();
  if(runs.runs.length) {
    await expect(page.locator('table tbody tr')).toHaveCount(runs.runs.length);
    await page.getByRole('button',{name:'Playground'}).click();
    await page.locator('#policy').selectOption(runs.runs[0].id);
    await page.waitForFunction(id=>window.arenaTelemetry.policy===id,runs.runs[0].id);
  } else {
    await expect(page.getByText(/No experiments yet/)).toBeVisible();
  }
});

test('mobile layout stays within the screen and guide opens', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await page.waitForFunction(()=>window.arenaTelemetry?.time>.1);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await expect(page.getByRole('button',{name:/Follow along/})).toBeVisible();
  await page.getByRole('button',{name:/Start an experiment/}).click();
  await expect(page.getByText('Teach. Measure. Repeat.')).toBeVisible();
  await page.getByRole('button',{name:'Close field guide'}).click();
  await page.screenshot({path:'../artifacts/screens/mobile.png',fullPage:true});
});

test('five simultaneous participants receive advancing independent simulations', async ({ browser }) => {
  const context = await browser.newContext();
  try {
    const pages = await Promise.all(Array.from({length:5},()=>context.newPage()));
    await Promise.all(pages.map(async p=>{
      await p.goto('http://127.0.0.1:8000/');
      await p.waitForFunction(()=>window.arenaTelemetry?.time>.2);
    }));
    await pages[0].getByRole('button',{name:/Find a treat/}).click();
    await pages[0].waitForFunction(()=>window.arenaTelemetry.behavior==='approach');
    expect(await pages[1].evaluate(()=>window.arenaTelemetry.behavior)).toBe('stop');
    const times=await Promise.all(pages.map(p=>p.evaluate(()=>window.arenaTelemetry.time)));
    await pages[0].waitForTimeout(300);
    for(let i=0;i<pages.length;i++)expect(await pages[i].evaluate(()=>window.arenaTelemetry.time)).toBeGreaterThan(times[i]);
  }finally{await context.close();}
});
