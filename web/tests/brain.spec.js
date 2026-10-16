import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { pcmWave } from '../src/audio.js';
import { recordPage } from '../scripts/capture.mjs';

const fixture = fs.readFileSync('../tests/assets/fetch-ball.wav');
const dataStart = fixture.indexOf(Buffer.from('data'));
const pcm = fixture.subarray(dataStart+8,dataStart+8+fixture.readUInt32LE(dataStart+4));
const speech = new Float32Array(pcm.length/2);
for(let i=0;i<speech.length;i++) speech[i]=pcm.readInt16LE(i*2)/32768;
const voicePath = path.resolve('../artifacts/fake-voice.wav');
fs.mkdirSync(path.dirname(voicePath),{recursive:true});
fs.writeFileSync(voicePath,Buffer.from(pcmWave([new Float32Array(8000),speech,new Float32Array(32000)],16000)));

test.use({launchOptions:{
  executablePath:process.env.CHROMIUM_PATH || (fs.existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined),
  args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',`--use-file-for-fake-audio-capture=${voicePath}`],
}});

test('RGB plan matrix responds to text and energy; hop has actual flight and landing',async({page})=>{
  test.setTimeout(60000);
  await page.goto('/');
  await page.waitForFunction(()=>window.arenaTelemetry?.time>.5);
  expect(await page.evaluate(()=>window.arenaTelemetry.brain)).toBe('neural');
  await expect(page.locator('#policy')).toHaveValue('baseline/curious-v2');
  await page.getByLabel('Talk to Miso').fill('fetch the ball');
  await page.getByRole('button',{name:/Send/}).click();
  await page.waitForFunction(()=>window.arenaTelemetry.planner?.plan?.rows[0].skill==='retrieve');
  await expect(page.locator('#plan-rows tbody tr')).toHaveCount(6);
  await page.waitForFunction(()=>document.querySelector('#sensor-camera').naturalWidth===96);
  await page.locator('#play-energy').evaluate(el=>{el.value='5';el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));});
  await page.waitForFunction(()=>window.arenaTelemetry.planner?.reason.includes('Resting to recharge'));
  await page.getByRole('button',{name:/Reset posture/}).click();
  await page.waitForFunction(()=>window.arenaTelemetry.time<1);
  await page.locator('[data-command="jump"]').click();
  await page.waitForFunction(()=>window.arenaTelemetry.missionComplete,{},{timeout:15000});
  const result=await page.evaluate(()=>window.arenaTelemetry);
  expect(result.jumpFlightSeconds).toBeGreaterThan(.1);
  expect(result.jumpMaxHeight).toBeGreaterThan(.43);
  expect(result.needsReset).toBe(false);
  await page.screenshot({path:'../artifacts/screens/arena-brain.png',fullPage:true});
});

test('real microphone capture/worklet/offline ASR produces reviewable transcript',async({page,context})=>{
  test.setTimeout(30000);
  await context.grantPermissions(['microphone']);
  await page.goto('/');
  await page.waitForFunction(()=>window.arenaTelemetry?.time>.1);
  await page.locator('#speak').click();
  await expect(page.locator('#speak')).toHaveAttribute('aria-pressed','true');
  await page.waitForTimeout(3200);
  await page.locator('#speak').click();
  await expect(page.getByLabel('Talk to Miso')).toHaveValue('fetch the ball',{timeout:10000});
  expect(await page.evaluate(()=>window.arenaTelemetry.missionComplete)).toBe(false);
  expect(await page.evaluate(()=>window.arenaTelemetry.behavior)).toBe('stop');
  await expect(page.locator('#toast')).toContainText('Press Send');
  await page.getByRole('button',{name:/Send/}).click();
  await page.waitForFunction(()=>window.arenaTelemetry.planner?.intent==='fetch');
});

test('browser RGB agent picks up, carries home and releases a physical ball',async({browser})=>{
  test.setTimeout(240000);
  const context=await browser.newContext({viewport:{width:1440,height:process.env.ROBOPUP_CAPTURE==='1'?1440:1050}});
  const page=await context.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let finishCapture;
  try{
    await page.goto('http://127.0.0.1:8000/');
    await page.waitForFunction(()=>window.arenaTelemetry?.time>.5);
    if(process.env.ROBOPUP_CAPTURE==='1')finishCapture=await recordPage(context,page,path.resolve('../artifacts/fetch-capture'));
    await page.locator('[data-command="fetch the ball"]').click();
    await page.waitForFunction(()=>window.arenaTelemetry.held,{},{timeout:110000});
    const carrying=await page.evaluate(()=>window.arenaTelemetry);
    expect(carrying.planner.ever_held).toBe(true);
    await page.screenshot({path:'../artifacts/screens/fetch-carry.png',fullPage:true});
    await page.waitForFunction(()=>window.arenaTelemetry.missionComplete,{},{timeout:110000});
    const result=await page.evaluate(()=>window.arenaTelemetry);
    expect(result.planner.maximum_ball_height).toBeGreaterThan(.2);
    expect(result.held).toBe(false);
    expect(result.needsReset).toBe(false);
    expect(result.planner.reason).toContain('contact, lift, return and release verified');
    expect(errors).toEqual([]);
    fs.writeFileSync('../artifacts/browser-fetch-result.json',JSON.stringify(result,null,2));
    await page.screenshot({path:'../artifacts/screens/fetch-complete.png',fullPage:true});
  }finally{
    if(finishCapture)await finishCapture();
    await context.close();
  }
});

test('microphone WAV resampling preserves duration and clips saturated samples',()=>{
  const wave=pcmWave([new Float32Array(48000).fill(1.5)],48000);
  const view=new DataView(wave);
  expect(view.getUint32(24,true)).toBe(16000);
  expect(view.getUint32(40,true)).toBe(32000);
  expect(view.getInt16(44,true)).toBe(32767);
  expect(wave.byteLength).toBe(32044);
});
