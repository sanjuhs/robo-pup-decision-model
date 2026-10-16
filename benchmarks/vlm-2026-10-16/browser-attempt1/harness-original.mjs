// Explicit real-model integration check. No mocked routes or model responses.
// Requires a running native backend with downloaded/configured VLM weights.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '@playwright/test';
import {recordPage} from './capture.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const output=path.resolve(process.env.ROBOPUP_VLM_BROWSER_OUTPUT || path.join(root,'artifacts/vlm-browser-real'));
const url=process.env.ROBOPUP_BROWSER_URL || 'http://127.0.0.1:8000';
fs.mkdirSync(output,{recursive:false});
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1440}});
if(process.env.ROBOPUP_ARENA_TOKEN)await context.addInitScript(token=>sessionStorage.setItem('robopup.accessKey',token),process.env.ROBOPUP_ARENA_TOKEN);
const page=await context.newPage();
const started=Date.now();
const record={schema:'robopup-genuine-vlm-browser-v1',success:false,model_response_fixture:false,physics:'native MuJoCo backend',renderer:'Chromium SwiftShader',instruction:'Fetch the red ball, then hop, then stop.',phases:[],responses:[],metrics:{ever_touched:false,ever_held:false,maximum_held_ball_height:0},errors:[]};
let latest,lastPhase,lastRaw,finishCapture,carryCaptured=false;
const screenshots=[];
page.on('pageerror',e=>record.errors.push(e.message));
page.on('websocket',socket=>socket.on('framereceived',frame=>{
  let state;try{state=JSON.parse(frame.payload);}catch{return;}
  if(state.type!=='state')return;
  latest=state;
  record.metrics.ever_touched ||= Boolean(state.ball_touched);
  record.metrics.ever_held ||= Boolean(state.held);
  if(state.held)record.metrics.maximum_held_ball_height=Math.max(record.metrics.maximum_held_ball_height,state.ball?.[2] || 0);
  const brain=state.planner;
  if(!brain || state.brain!=='vlm')return;
  const phase=`${brain.phase}/${brain.current_step}`;
  if(phase!==lastPhase){
    record.phases.push({phase:brain.phase,current_step:brain.current_step,wall_seconds:(Date.now()-started)/1000,sim_seconds:state.time});
    console.log(JSON.stringify(record.phases.at(-1)));lastPhase=phase;
  }
  if(brain.raw_response && brain.raw_response!==lastRaw){
    record.responses.push({raw_text:brain.raw_response,model:brain.model,provider:brain.provider,inference_ms:brain.inference_ms,current_step:brain.current_step});
    lastRaw=brain.raw_response;
  }
  if(state.held && !carryCaptured){carryCaptured=true;screenshots.push(page.screenshot({path:path.join(output,'carrying.png'),fullPage:true}));}
}));
try{
  await page.goto(url);
  await page.waitForFunction(()=>window.arenaTelemetry?.time>.5);
  const status=await (await page.request.get(`${url}/api/vlm`)).json();
  assert.equal(status.available,true,'A real configured VLM is required; this check does not supply a fixture');
  record.initial_runtime=status;
  await page.getByRole('button',{name:/Close-up/}).click();
  await page.getByLabel('Decision brain').selectOption('vlm');
  await page.waitForFunction(()=>window.arenaTelemetry?.brain==='vlm');
  if(process.env.ROBOPUP_CAPTURE==='1')finishCapture=await recordPage(context,page,path.join(output,'capture'));
  await page.getByLabel('Talk to Miso').fill(record.instruction);
  await page.getByRole('button',{name:/Send/}).click();
  await page.waitForFunction(()=>document.querySelector('#sensor-camera').naturalWidth===384,{},{timeout:20000});
  await page.screenshot({path:path.join(output,'thinking.png'),fullPage:true});
  await page.waitForFunction(()=>window.arenaTelemetry?.planner?.complete || window.arenaTelemetry?.planner?.failed,{},{timeout:600000});
  const final=structuredClone(latest);
  if(final.planner?.camera_png){fs.writeFileSync(path.join(output,'last-eye.png'),Buffer.from(final.planner.camera_png,'base64'));delete final.planner.camera_png;}
  record.final=final;
  assert.equal(final.brain,'vlm');
  assert.equal(final.planner.model,'Qwen/Qwen3-VL-2B-Instruct');
  assert.equal(final.planner.complete,true,final.planner.reason);
  assert.equal(final.mission_complete,true);
  assert.equal(final.needs_reset,false);
  assert.equal(final.held,false);
  assert.ok(record.metrics.ever_touched && record.metrics.ever_held);
  assert.ok(record.metrics.maximum_held_ball_height>.2);
  assert.ok(Math.hypot(...final.ball.slice(0,2))<.6 && final.ball[2]<.19);
  assert.ok(final.jump_flight_seconds>.04 && final.foot_contacts.length>=2 && final.upright>.9);
  assert.deepEqual(final.planner.mission_steps.map(s=>s.action),['fetch','jump','stop']);
  assert.ok(final.planner.mission_steps.every(s=>s.status==='complete'));
  assert.ok(record.responses.length>=3);
  assert.deepEqual(record.errors,[]);
  await page.getByText('Inspect actual model response').click();
  await page.screenshot({path:path.join(output,'complete.png'),fullPage:true});
  record.success=true;
}catch(error){record.error=error.message;try{await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});}catch{}}
finally{
  await Promise.allSettled(screenshots);
  if(finishCapture)try{await finishCapture();record.capture={source:'actual CDP frames',directory:'capture',cropped_or_sped_up:false};}catch(error){record.recording_error=error.message;}
  record.wall_seconds=(Date.now()-started)/1000;
  fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(record,null,2)+'\n');
  await browser.close();
}
console.log(JSON.stringify({output,success:record.success,wall_seconds:record.wall_seconds}));
if(!record.success)process.exitCode=1;
