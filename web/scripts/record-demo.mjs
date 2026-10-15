// Run against a built, running local arena: node scripts/record-demo.mjs
// Records real browser playback; never starts training or paid infrastructure.
import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../',import.meta.url));
const out = path.join(root,'artifacts/presentation', new Date().toISOString().replaceAll(':','-'));
fs.mkdirSync(out, {recursive:true});
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || (fs.existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined),
  args: ['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
});
try {
  const context = await browser.newContext({
    viewport:{width:1440,height:1050},
  });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8000/');
  await page.waitForFunction(()=>window.arenaPresentation?.time>.1);
  await page.locator('#policy').selectOption('runpod/5090-once');
  await page.waitForFunction(()=>window.arenaTelemetry.policy==='runpod/5090-once');
  await page.getByRole('button',{name:/Close-up/}).click();
  await page.waitForFunction(()=>window.arenaPresentation.cameraDistance<2.7);
  const viewport = await page.locator('#viewport').boundingBox();
  // CDP frames use the installed system FFmpeg, avoiding a separate video binary.
  const cdp = await context.newCDPSession(page);
  const captured = [];
  cdp.on('Page.screencastFrame', async frame => {
    const file = `frame-${String(captured.length).padStart(6,'0')}.jpg`;
    fs.writeFileSync(path.join(out,file),Buffer.from(frame.data,'base64'));
    captured.push({file,time:frame.metadata.timestamp});
    await cdp.send('Page.screencastFrameAck',{sessionId:frame.sessionId});
  });
  await cdp.send('Page.startScreencast',{format:'jpeg',quality:85,maxWidth:1440,maxHeight:1050});
  const wallStart = Date.now();
  const start = await page.evaluate(()=>window.arenaTelemetry);
  await page.evaluate(()=>{
    window.demoFrames=[];
    const collect=()=>{
      window.demoFrames.push({wall:performance.now(),...window.arenaPresentation});
      if(window.demoFrames.length<2000)window.demoRequest=requestAnimationFrame(collect);
    };
    window.demoRequest=requestAnimationFrame(collect);
  });
  await page.keyboard.down('w');
  await page.waitForTimeout(6000);
  const end = await page.evaluate(()=>window.arenaTelemetry);
  const wallSeconds=(Date.now()-wallStart)/1000;
  await cdp.send('Page.stopScreencast');
  const frames = await page.evaluate(()=>{cancelAnimationFrame(window.demoRequest);return window.demoFrames;});
  await page.screenshot({path:path.join(out,'walking.png'),fullPage:true});
  await page.keyboard.up('w');
  const gaps=frames.slice(1).map((f,i)=>f.wall-frames[i].wall).sort((a,b)=>a-b);
  fs.writeFileSync(path.join(out,'measurements.json'),JSON.stringify({
    policy:end.policy,renderer:'Chromium headless / SwiftShader software WebGL',
    viewport,wallSeconds,simulationSeconds:end.time-start.time,
    worldXDisplacement:end.position[0]-start.position[0],needsReset:end.needsReset,
    frameCount:frames.length,medianFrameMs:gaps[Math.floor(gaps.length*.5)],
    p95FrameMs:gaps[Math.floor(gaps.length*.95)],frames,
  },null,2)+'\n');
  await context.close();
  if(captured.length<2)throw new Error('Browser did not produce enough video frames');
  const concat = captured.map((f,i)=>`file '${f.file}'\nduration ${Math.max(.001,(captured[i+1]?.time??f.time+.04)-f.time)}`).join('\n');
  fs.writeFileSync(path.join(out,'frames.ffconcat'),concat+'\n');
  execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',path.join(out,'frames.ffconcat'),'-c:v','libvpx-vp9','-crf','34','-b:v','0',path.join(out,'walking.webm')]);
  console.log('Saved browser recording, screenshot and measured telemetry: '+out);
} finally {
  await browser.close();
}
