import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import './style.css';
import './commands.css';

const $ = (id) => document.getElementById(id);
let socket, state, paused = false, followCamera = false, manual = false, selectedMode = 'stop';
let toastTimer, lastPointer;
const keys = new Set();
const meshes = new Map();
const targetPos = new THREE.Vector3(2, 1, .02);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#142823');
scene.fog = new THREE.Fog('#142823', 10, 25);
const camera = new THREE.PerspectiveCamera(37, 1, .05, 70);
camera.up.set(0, 0, 1);
camera.position.set(3.0, -4.0, 3.3);
let renderer, controls;
try {
  renderer = new THREE.WebGLRenderer({ canvas: $('arena-canvas'), antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(.45, .1, .15);
  controls.enableDamping = true;
  controls.dampingFactor = .07;
  controls.minDistance = 1.5;
  controls.maxDistance = 15;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.update();
} catch (error) {
  $('renderer-error').hidden = false;
  $('connection-text').textContent = 'WebGL unavailable';
}

scene.add(new THREE.AmbientLight('#e7f8e2', 1.5));
const sun = new THREE.DirectionalLight('#fff4d4', 3.6);
sun.position.set(-2, -4, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -6;
sun.shadow.camera.right = sun.shadow.camera.top = 6;
sun.shadow.normalBias = .015;
sun.shadow.bias = -.0002;
scene.add(sun);
const rim = new THREE.DirectionalLight('#9fead6', 2);
rim.position.set(2, 5, 3);
scene.add(rim);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), new THREE.MeshStandardMaterial({ color: '#1f3930', roughness: 1 }));
ground.position.z = -.003;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(24, 48, '#47624e', '#2a4839');
grid.rotation.x = Math.PI / 2;
grid.position.z = .001;
grid.material.transparent = true;
grid.material.opacity = .45;
scene.add(grid);

const boundary = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-5, -5, .005), new THREE.Vector3(5, -5, .005),
  new THREE.Vector3(5, 5, .005), new THREE.Vector3(-5, 5, .005),
]), new THREE.LineBasicMaterial({ color: '#537553', transparent: true, opacity: .6 }));
scene.add(boundary);
for (const [x, y] of [[-5,-5],[-5,5],[5,-5],[5,5]]) {
  const marker = new THREE.Mesh(new THREE.BoxGeometry(.055, .055, .14), new THREE.MeshStandardMaterial({ color: '#9aac7e', roughness: .7 }));
  marker.position.set(x, y, .07);
  scene.add(marker);
}

// The orange ring marks the push goal. It does not exert any physical force.
const goal = new THREE.Mesh(new THREE.RingGeometry(.33, .35, 48), new THREE.MeshBasicMaterial({ color: '#ca9563', transparent: true, opacity: .6, side: THREE.DoubleSide }));
goal.position.set(3, 0, .007);
scene.add(goal);
const treat = new THREE.Group();
const ring = new THREE.Mesh(new THREE.TorusGeometry(.15, .009, 8, 48), new THREE.MeshBasicMaterial({ color: '#c7f697' }));
treat.add(ring);
const treatCore = new THREE.Mesh(new THREE.OctahedronGeometry(.065), new THREE.MeshStandardMaterial({ color: '#d8f8a6', emissive: '#a1c567', emissiveIntensity: .4, roughness: .3 }));
treatCore.position.z = .18;
treat.add(treatCore);
scene.add(treat);

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}

function material(color, name) {
  const c = new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);
  return new THREE.MeshStandardMaterial({ color: c, roughness: .5, metalness: name?.includes('foot') ? .1 : .15 });
}

function buildScene(geoms) {
  for (const geom of geoms) {
    const [a, b, c] = geom.size;
    let geometry;
    if (geom.type === 'box') geometry = new RoundedBoxGeometry(a*2, b*2, c*2, 3, Math.min(a,b,c)*.2);
    else if (geom.type === 'sphere') geometry = new THREE.SphereGeometry(a, 20, 14);
    else if (geom.type === 'capsule') {
      geometry = new THREE.CapsuleGeometry(a, b*2, 5, 12);
      geometry.rotateX(Math.PI/2);
    } else {
      geometry = new THREE.CylinderGeometry(a, a, b*2, 24);
      geometry.rotateX(Math.PI/2);
    }
    const mesh = new THREE.Mesh(geometry, material(geom.color, geom.name));
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
    meshes.set(geom.id, {mesh, target: new THREE.Vector3(), quaternion: new THREE.Quaternion(), initialized: false});
    if (geom.name === 'head') {
      for (const side of [-1,1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(.013, 12, 8), new THREE.MeshStandardMaterial({ color: '#122b27', roughness: .3 }));
        eye.position.set(.091, side*.052, .018);
        mesh.add(eye);
        const glint = new THREE.Mesh(new THREE.SphereGeometry(.004, 8, 8), new THREE.MeshBasicMaterial({ color:'#e7ffd2' }));
        glint.position.set(.101, side*.052-.004, .023);
        mesh.add(glint);
        const ear = new THREE.Mesh(new RoundedBoxGeometry(.055, .035, .07, 2, .01), new THREE.MeshStandardMaterial({color:'#60765c'}));
        ear.position.set(-.035, side*.078, .09);
        ear.rotation.x = side*.2;
        ear.castShadow = true;
        mesh.add(ear);
      }
      const nose = new THREE.Mesh(new THREE.BoxGeometry(.018,.026,.02),new THREE.MeshStandardMaterial({color:'#29483b'}));
      nose.position.set(.095,0,-.015);
      mesh.add(nose);
    }
    if (geom.name === 'torso_geom') {
      const panel = new THREE.Mesh(new RoundedBoxGeometry(.2,.16,.012,2,.006),new THREE.MeshStandardMaterial({color:'#c4e5b4',roughness:.6}));
      panel.position.set(-.02,0,.078);
      mesh.add(panel);
      const tail = new THREE.Mesh(new THREE.CapsuleGeometry(.017,.15,3,8),new THREE.MeshStandardMaterial({color:'#b5d1a4'}));
      tail.rotation.z = Math.PI/2;
      tail.rotation.y = -.5;
      tail.position.set(-.33,0,.05);
      tail.castShadow = true;
      mesh.add(tail);
    }
    if (geom.name === 'crate_geom') {
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(a*2+.002,b*2+.002,c*2+.002)),new THREE.LineBasicMaterial({color:'#744f29',transparent:true,opacity:.6}));
      mesh.add(edges);
      const tape = new THREE.Mesh(new THREE.BoxGeometry(a*2+.003,.035,c*2+.003),new THREE.MeshStandardMaterial({color:'#e3b479',roughness:.9}));
      mesh.add(tape);
    }
    if (geom.name === 'ball_geom') {
      for (const axis of [0,1,2]) {
        const band = new THREE.Mesh(new THREE.TorusGeometry(a+.001,.005,6,32),new THREE.MeshStandardMaterial({color:'#ffe6bd',roughness:.8}));
        if(axis===1)band.rotation.x=Math.PI/2;
        if(axis===2)band.rotation.y=Math.PI/2;
        mesh.add(band);
      }
    }
  }
}

const rotationMatrix = new THREE.Matrix4();
function updateState(next) {
  state = next;
  $('sim-time').textContent = `${next.time.toFixed(1)} s`;
  $('speed').innerHTML = `${next.speed.toFixed(2)} <small>m/s</small>`;
  const balance = Math.round(Math.max(0,next.upright)*100);
  $('balance').innerHTML = `${balance} <small>%</small>`;
  $('speed-meter').style.width = `${Math.min(100,next.speed/.6*100)}%`;
  $('balance-meter').style.width = `${balance}%`;
  $('thought').textContent = next.mission_complete ? 'Mission complete. Measured in the physics world.' : next.reason;
  $('pet-state').textContent = next.needs_reset ? 'A little tumble. Reset posture?' : next.paused ? 'Taking a breather' : ({stop:'Standing by. Ready for a treat.',approach:'On a little adventure',follow:'Following my curiosity',push:'Testing a little nudge',manual:'You lead. I’ll try.',recover:'Finding my feet'}[next.behavior]);
  $('pet-status')?.classList.toggle('fallen',next.needs_reset);
  targetPos.set(next.target[0],next.target[1],.015);
  for (const geom of next.geoms) {
    const item = meshes.get(geom.id);
    if (!item) continue;
    item.target.fromArray(geom.position);
    const r = geom.rotation;
    rotationMatrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
    item.quaternion.setFromRotationMatrix(rotationMatrix);
    if (!item.initialized) {
      item.mesh.position.copy(item.target);
      item.mesh.quaternion.copy(item.quaternion);
      item.initialized = true;
    }
  }
  // Expose only non-sensitive simulation telemetry for end-to-end verification.
  window.arenaTelemetry = { time: next.time, behavior: next.behavior, needsReset:next.needs_reset, geometryCount: next.geoms.length, policy:next.policy, missionComplete:next.mission_complete, ballTouched:next.ball_touched };
}

function send(data) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(data));
}

function chooseMode(mode) {
  manual = mode === 'manual';
  selectedMode = mode;
  for (const button of document.querySelectorAll('[data-mode]')) button.classList.toggle('selected',button.dataset.mode === mode);
  if (!manual) keys.clear();
  send({mode});
}

function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.onopen = () => {
    $('connection-dot').classList.add('connected');
    $('connection-text').textContent = 'Arena connected';
  };
  socket.onmessage = ({data}) => {
    const message = JSON.parse(data);
    if (message.type === 'scene') buildScene(message.geoms);
    else if (message.type === 'state') updateState(message);
    else if (message.type === 'error') toast(message.message);
    else if (message.type === 'reply') toast(message.message);
  };
  socket.onclose = () => {
    $('connection-dot').classList.remove('connected');
    $('connection-text').textContent = 'Disconnected · reload to reconnect';
    toast('Arena disconnected. Check the server and reload to reconnect.');
  };
}

const resize = () => {
  if (!renderer) return;
  const {width,height} = $('viewport').getBoundingClientRect();
  renderer.setSize(width,height,false);
  camera.aspect = width/Math.max(1,height);
  camera.updateProjectionMatrix();
};
new ResizeObserver(resize).observe($('viewport'));
const projected = new THREE.Vector3();
function animate(time) {
  requestAnimationFrame(animate);
  if (!renderer) return;
  for (const item of meshes.values()) {
    item.mesh.position.lerp(item.target,.4);
    item.mesh.quaternion.slerp(item.quaternion,.4);
  }
  treat.position.lerp(targetPos,.15);
  treatCore.position.z = .18 + Math.sin(time*.003)*.035;
  treatCore.rotation.z = time*.0006;
  ring.scale.setScalar(1+Math.sin(time*.003)*.04);
  if (state) {
    projected.set(state.position[0],state.position[1],state.position[2]+.34);
    if (followCamera) {
      const desired = new THREE.Vector3(state.position[0],state.position[1],.2);
      const shift = desired.clone().sub(controls.target).multiplyScalar(.035);
      controls.target.add(shift);
      camera.position.add(shift);
    }
    projected.project(camera);
    const rect = $('viewport').getBoundingClientRect();
    $('world-label').style.left = `${(projected.x*.5+.5)*rect.width}px`;
    $('world-label').style.top = `${(-projected.y*.5+.5)*rect.height+$('viewport').offsetTop}px`;
    $('world-label').style.opacity = Math.abs(projected.x)>1 || Math.abs(projected.y)>1 ? '0' : '1';
  }
  controls.update();
  renderer.render(scene,camera);
}
requestAnimationFrame(animate);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const plane = new THREE.Plane(new THREE.Vector3(0,0,1),0);
const hit = new THREE.Vector3();
$('viewport').addEventListener('pointerdown',(event)=>{ lastPointer=[event.clientX,event.clientY]; });
$('viewport').addEventListener('pointerup',(event)=>{
  if (!renderer || !lastPointer || Math.hypot(event.clientX-lastPointer[0],event.clientY-lastPointer[1])>5) return;
  const rect = $('viewport').getBoundingClientRect();
  pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
  raycaster.setFromCamera(pointer,camera);
  if (raycaster.ray.intersectPlane(plane,hit)) {
    send({target:[THREE.MathUtils.clamp(hit.x,-5,5),THREE.MathUtils.clamp(hit.y,-5,5)]});
    if (selectedMode === 'follow') chooseMode('approach');
    toast('Treat dropped. Ask Miso to find it.');
  }
});
for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click',()=>chooseMode(button.dataset.mode));
$('reset').onclick = () => {chooseMode('stop');send({reset:true});toast('Posture restored. This resets the simulation; self-righting is not learned.');};
$('pause').onclick = () => { paused=!paused;send({paused});$('pause').textContent=paused?'▶ Resume':'Ⅱ Pause';$('pause').setAttribute('aria-label',paused?'Resume simulation':'Pause simulation'); };
$('camera').onclick = () => {followCamera=!followCamera;$('camera').classList.toggle('active',followCamera);};
$('policy').onchange = () => {send({policy:$('policy').value});$('policy-detail').textContent=$('policy').value==='gait'?'Rule-based decisions + an analytic gait. Select a trained policy to compare.':'Fresh residual PPO weights. Flat forward training only; other commands are experimental.';};
$('command-form').onsubmit = (event) => {
  event.preventDefault();
  const text=$('pet-command').value.trim();
  if(!text)return;
  manual=false;keys.clear();selectedMode='approach';
  send({text});$('pet-command').value='';
};

window.addEventListener('keydown',(event)=>{
  if (['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (['w','a','s','d',' '].includes(key)) {
    event.preventDefault();
    if (key===' ') {chooseMode('stop');return;}
    keys.add(key);
    if (!manual) chooseMode('manual');
    sendManual();
  }
});
window.addEventListener('keyup',(event)=>{keys.delete(event.key.toLowerCase());if(manual)sendManual();});
window.addEventListener('blur',()=>{keys.clear();if(manual)sendManual();});
function sendManual() {send({mode:'manual',forward:(keys.has('w')?.35:0)-(keys.has('s')?.25:0),turn:(keys.has('a')?1:0)-(keys.has('d')?1:0)});}
setInterval(()=>{if(manual)sendManual();},100);

function showPage(page) {
  $('playground').hidden=page!=='arena';
  document.querySelector('.under-grid').hidden=page!=='arena';
  $('experiments').hidden=page!=='experiments';
  $('arena-tab').classList.toggle('nav-active',page==='arena');
  $('experiments-tab').classList.toggle('nav-active',page==='experiments');
  if(page==='experiments')refreshRuns();
  else resize();
}
$('arena-tab').onclick=()=>showPage('arena');
$('experiments-tab').onclick=()=>showPage('experiments');
for(const id of ['guide-tab','training-guide'])$(id).onclick=()=>{$('guide').hidden=false;};
$('close-guide').onclick=()=>{$('guide').hidden=true;};
$('refresh-runs').onclick=refreshRuns;

async function refreshRuns() {
  try {
    const response=await fetch('/api/runs');
    if(!response.ok)throw new Error('Results unavailable');
    const {runs}=await response.json();
    $('run-count').textContent=runs.length;
    const previous=$('policy').value;
    $('policy').replaceChildren(new Option('Analytic trot · no learned weights','gait'));
    for(const run of runs)if(run.has_policy)$('policy').add(new Option(`${run.participant} / ${run.run}`,run.id));
    if([...$('policy').options].some(o=>o.value===previous))$('policy').value=previous;
    const table=document.createElement('table');
    const header=table.createTHead().insertRow();
    for(const title of ['Participant / run','Steps','Train time','Speed / target','Return','Falls']){const th=document.createElement('th');th.textContent=title;header.append(th);}
    const body=table.createTBody();
    for(const run of runs){const row=body.insertRow();for(const text of [`${run.participant} / ${run.run}`,run.steps.toLocaleString(),`${run.seconds.toFixed(1)} s`,`${run.speed.toFixed(3)} / ${run.target_speed.toFixed(2)} m/s`,run.return.toFixed(1),`${(run.fall_rate*100).toFixed(0)}%`])row.insertCell().textContent=text;}
    $('runs-table').replaceChildren();
    if(runs.length)$('runs-table').append(table);
    else{const empty=document.createElement('p');empty.className='empty';empty.textContent='No experiments yet. Every pup starts somewhere. Train a policy with the command below, then refresh.';$('runs-table').append(empty);}
  }catch(error){toast('Could not load experiments. Check the arena server.');}
}

connect();
refreshRuns();
