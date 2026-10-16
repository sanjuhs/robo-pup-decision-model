import {test,expect} from '@playwright/test';
import {backendEndpoints,readBackendOverride,readSessionToken} from '../src/backend.js';

test('external backend paths preserve HTTPS/WSS and never accept credentials in a URL',()=>{
  const remote=backendEndpoints('https://robot.example/arena/','http://localhost:8000');
  expect(remote.base).toBe('https://robot.example/arena');
  expect(remote.api('transcribe')).toBe('https://robot.example/arena/api/transcribe');
  expect(remote.ws).toBe('wss://robot.example/arena/ws');
  expect(backendEndpoints('','http://localhost:8000').ws).toBe('ws://localhost:8000/ws');
  for(const bad of ['https://user:secret@robot.example','https://robot.example?token=secret','ftp://robot.example','https://robot.example#key'])expect(()=>backendEndpoints(bad)).toThrow();
  const unavailable={getItem:()=>{throw new Error('Storage disabled');}};
  expect(readBackendOverride(unavailable)).toBe('');
  expect(readSessionToken(unavailable)).toBe('');
});

test('static frontend routes APIs and first-message workshop authentication to its selected backend',async({page})=>{
  // UI transport fixture, not evidence of model inference or physical execution.
  const origin='https://arena-backend.example';
  const token='workshop-test-fixture';
  const requests=[];const messages=[];
  await page.addInitScript(({origin,token})=>{
    localStorage.setItem('robopup.backendUrl',origin);
    sessionStorage.setItem('robopup.accessKey',token);
  },{origin,token});
  await page.route(`${origin}/api/**`,route=>{
    requests.push({url:route.request().url(),authorization:route.request().headers().authorization});
    const path=new URL(route.request().url()).pathname;
    return route.fulfill({json:path.endsWith('/runs')?{runs:[]}:path.endsWith('/vlm')?{available:true,provider:'UI contract fixture',model:'fixture/test-vlm',status:'ready'}:{status:'ok'}});
  });
  await page.routeWebSocket(`${origin.replace('https:','wss:')}/ws`,ws=>{
    ws.onMessage(message=>{
      messages.push(JSON.parse(message));
      if(JSON.parse(message).brain==='vlm')ws.send(JSON.stringify({
        type:'state',time:1,speed:0,upright:1,paused:false,needs_reset:false,reason:'Fixture observation',target:[0,0],position:[0,0,.3],geoms:[],policy:'Analytic gait',behavior:'stop',brain:'vlm',held:false,energy:1,
        planner:{model:'fixture/test-vlm',provider:'UI contract fixture',phase:'executing',reason:'UI fixture: execute a bounded sequence',elapsed_s:1,inference_ms:150,current_step:1,
          camera_png:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jkAAAAABJRU5ErkJggg==',
          mission_steps:[{action:'approach',object:'ball',status:'complete'},{action:'stop',object:'none',status:'running'}],
          eye_grounding:{visible_objects:['ball'],objects:{ball:{visible:true}},limitation:'UI fixture: limited arena color check.'},
          raw_response:'{"steps":[{"action":"approach","object":"ball"},{"action":"stop","object":"none"}]}',limitation:'UI fixture only; no model or physics claim.'},
      }));
    });
  });
  await page.goto('/');
  await expect(page.getByText('Arena connected')).toBeVisible();
  await page.getByLabel('Decision brain').selectOption('vlm');
  await expect(page.locator('#planner-model')).toHaveText('fixture/test-vlm');
  await expect(page.locator('#planner-phase')).toHaveText('executing');
  await expect(page.locator('#sensor-caption')).toContainText('first person');
  await expect(page.locator('#plan-rows tbody tr')).toHaveCount(2);
  await expect(page.locator('#plan-rows tbody tr').nth(1)).toHaveAttribute('data-status','running');
  await expect(page.locator('#plan-rows tbody tr').nth(1).locator('td').first()).toHaveCSS('background-color','rgb(231, 241, 217)');
  await page.getByText('Inspect actual model response').click();
  await expect(page.locator('#vlm-raw-response')).toContainText('"steps"');
  await expect(page.locator('#vlm-limitation')).toContainText('UI fixture only');
  await expect(page.locator('#vlm-grounding')).toContainText('Activation eye check: ball.');
  await expect(page.locator('#vlm-grounding')).toContainText('UI fixture: limited arena color check.');
  expect(messages[0]).toEqual({type:'auth',token});
  expect(requests.find(request=>request.url.endsWith('/api/runs')).authorization).toBe(`Bearer ${token}`);
  expect(requests.find(request=>request.url.endsWith('/api/health')).authorization).toBeUndefined();
  expect(requests.every(request=>!request.url.includes(token))).toBe(true);
  expect(await page.evaluate(()=>window.arenaConnection.backend)).toBe(origin);
});

test('unreachable static deployment shows a backend setup panel instead of claiming an active arena',async({page})=>{
  await page.routeWebSocket('**/ws',ws=>ws.close({code:1011,reason:'Offline UI fixture'}));
  await page.goto('/');
  await expect(page.getByRole('region',{name:'Connect physics backend'})).toBeVisible();
  await expect(page.locator('#backend-message')).toContainText('disconnected');
  await expect(page.locator('#connection-text')).toContainText('Disconnected');
  await page.getByLabel('Workshop access key · optional').fill('test-session-key');
  expect(await page.locator('#backend-token').getAttribute('type')).toBe('password');
});
