// Explicit real-model integration check. No mocked routes or model responses.
// Requires a running native backend with downloaded/configured VLM weights.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
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
const record={schema:'robopup-genuine-vlm-browser-v2',success:false,model_response_fixture:false,physics:'native MuJoCo backend',renderer:'Chromium SwiftShader',instruction:'Fetch the red ball, then hop, then stop.',phases:[],responses:[],metrics:{ever_touched:false,ever_held:false,maximum_held_ball_height:0},physical_events:[],step_completions:[],errors:[]};
try {
  record.source_ref=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
  try {execFileSync('git',['diff','--quiet','HEAD','--','src','web','scripts','services','configs','demo','pyproject.toml','uv.lock','vercel.json'],{cwd:root});record.runtime_source_clean=true;}
  catch {record.runtime_source_clean=false;}
} catch {record.source_ref=null;record.runtime_source_clean=false;}
record.harness_sha256=createHash('sha256').update(fs.readFileSync(fileURLToPath(import.meta.url))).digest('hex');
let latest,lastPhase,lastRaw,finishCapture,missionStarted,carryCaptured=false,previouslyHeld=false;
const completedSteps=new Set();
function physicalSnapshot(state) {
  return {
    wall_seconds:(Date.now()-started)/1000,sim_seconds:state.time,
    ball:state.ball?.slice(),position:state.position?.slice(),held:Boolean(state.held),
    ball_touched:Boolean(state.ball_touched),upright:state.upright,
    foot_contacts:state.foot_contacts?.slice(),needs_reset:Boolean(state.needs_reset),
    jump_flight_seconds:state.jump_flight_seconds,jump_max_height:state.jump_max_height,
  };
}
const screenshots=[];
page.on('pageerror',e=>record.errors.push(e.message));
page.on('websocket',socket=>socket.on('framereceived',frame=>{
  let state;try{state=JSON.parse(frame.payload);}catch{return;}
  if(state.type!=='state')return;
  latest=state;
  record.metrics.ever_touched ||= Boolean(state.ball_touched);
  record.metrics.ever_held ||= Boolean(state.held);
  if(state.held)record.metrics.maximum_held_ball_height=Math.max(record.metrics.maximum_held_ball_height,state.ball?.[2] || 0);
  if(Boolean(state.held)!==previouslyHeld) {
    record.physical_events.push({event:state.held?'grasp_contact':'release',...physicalSnapshot(state)});
    previouslyHeld=Boolean(state.held);
  }
  const brain=state.planner;
  if(!brain || state.brain!=='vlm')return;
  for(const [index,step] of (brain.mission_steps??[]).entries()) {
    if(step.status!=='complete' || completedSteps.has(index))continue;
    completedSteps.add(index);
    record.step_completions.push({index,action:step.action,object:step.object,...physicalSnapshot(state)});
  }
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
  const backend=await page.evaluate(()=>window.arenaConnection.backend);
  const status=await (await page.request.get(`${backend}/api/vlm`)).json();
  assert.equal(status.available,true,'A real configured VLM is required; this check does not supply a fixture');
  record.initial_runtime=status;
  await page.getByRole('button',{name:/Close-up/}).click();
  await page.getByLabel('Decision brain').selectOption('vlm');
  await page.waitForFunction(()=>window.arenaTelemetry?.brain==='vlm');
  if(process.env.ROBOPUP_CAPTURE==='1')finishCapture=await recordPage(context,page,path.join(output,'capture'));
  await page.getByLabel('Talk to Miso').fill(record.instruction);
  missionStarted=Date.now();
  record.setup_wall_seconds=(missionStarted-started)/1000;
  await page.getByRole('button',{name:/Send/}).click();
  await page.waitForFunction(()=>document.querySelector('#sensor-camera').naturalWidth===384,{},{timeout:20000});
  await page.screenshot({path:path.join(output,'thinking.png'),fullPage:true});
  await page.waitForFunction(()=>window.arenaTelemetry?.planner?.complete || window.arenaTelemetry?.planner?.failed,{},{timeout:900000});
  record.mission_wall_seconds=(Date.now()-missionStarted)/1000;
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
  // Verify fetch when it finishes. A later physical hop can legitimately nudge
  // an already released ball; final ball position is a separate reported metric.
  const fetchCompletion=record.step_completions.find(step=>step.action==='fetch' && step.object==='ball');
  assert.ok(fetchCompletion,'A fetch-completion frame must be observed independently');
  assert.equal(fetchCompletion.held,false);
  assert.equal(fetchCompletion.needs_reset,false);
  assert.ok(fetchCompletion.ball?.length===3 && Math.hypot(...fetchCompletion.ball.slice(0,2))<.6 && fetchCompletion.ball[2]<.19,'The released ball must be grounded near home at fetch completion');
  assert.ok(record.physical_events.some(event=>event.event==='release' && event.sim_seconds<=fetchCompletion.sim_seconds),'A real held-to-released transition must precede fetch completion');
  record.metrics.ball_home_distance_at_fetch_completion=Math.hypot(...fetchCompletion.ball.slice(0,2));
  record.metrics.ball_home_distance_after_hop=Math.hypot(...final.ball.slice(0,2));
  const jumpCompletion=record.step_completions.find(step=>step.action==='jump');
  assert.ok(jumpCompletion && jumpCompletion.jump_flight_seconds>.04 && jumpCompletion.foot_contacts.length>=2 && jumpCompletion.upright>.9,'Hop completion must include measured flight and stable ground contact');
  assert.ok(final.jump_flight_seconds>.04 && final.foot_contacts.length>=2 && final.upright>.9);
  assert.deepEqual(final.planner.mission_steps.map(s=>s.action),['fetch','jump','stop']);
  assert.ok(final.planner.mission_steps.every(s=>s.status==='complete'));
  assert.ok(record.responses.length>=3);
  assert.deepEqual(record.errors,[]);
  await page.getByText('Inspect actual model response').click();
  await page.screenshot({path:path.join(output,'complete.png'),fullPage:true});
  record.success=true;
}catch(error){record.error=error.message;if(missionStarted && record.mission_wall_seconds===undefined)record.mission_wall_seconds=(Date.now()-missionStarted)/1000;if(latest && !record.final){record.final=structuredClone(latest);if(record.final.planner)delete record.final.planner.camera_png;}try{await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});}catch{}}
finally{
  await Promise.allSettled(screenshots);
  const finalizationStarted=Date.now();
  if(finishCapture)try{await finishCapture();record.capture={source:'actual CDP frames',directory:'capture',cropped_or_sped_up:false,encoding:'VP9 cpu-used4 row-mt1; original frame timestamps retained'};}catch(error){record.recording_error=error.message;}
  record.capture_finalization_wall_seconds=(Date.now()-finalizationStarted)/1000;
  record.wall_seconds=(Date.now()-started)/1000;
  record.wall_seconds_includes_video_encoding=Boolean(finishCapture);
  fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(record,null,2)+'\n');
  await browser.close();
}
console.log(JSON.stringify({output,success:record.success,wall_seconds:record.wall_seconds}));
if(!record.success)process.exitCode=1;
