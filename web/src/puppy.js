import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// Original procedural artwork: no external model, texture, rig, or license fetch.
// Each part is attached to one physical geometry; the motor pose is never replaced.
export class PuppySkin {
  constructor() {
    this.parts = new Map();
    this.eyes = [];
    this.ears = [];
    this.sphere = new THREE.SphereGeometry(1, 24, 16);
    this.detailSphere = new THREE.SphereGeometry(1, 12, 8);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#b6b6b6';
    ctx.fillRect(0,0,256,256);
    let seed = 431;
    const random = () => { seed = (1664525*seed+1013904223)>>>0; return seed/4294967296; };
    for(let i=0;i<7000;i++) {
      const x=random()*256,y=random()*256,v=Math.floor(90+random()*100);
      ctx.strokeStyle=`rgb(${v},${v},${v})`;
      ctx.lineWidth=.5;
      ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+random()*2-1,y+2+random()*4);ctx.stroke();
    }
    this.fur = new THREE.CanvasTexture(canvas);
    this.fur.wrapS = this.fur.wrapT = THREE.RepeatWrapping;
    this.fur.repeat.set(3,2);
    const coat = color => new THREE.MeshStandardMaterial({color,roughness:.91,metalness:0,bumpMap:this.fur,bumpScale:.0018});
    this.gold=coat('#b78348');
    this.honey=coat('#8f5b30');
    this.cream=coat('#eed8ae');
    this.black=new THREE.MeshStandardMaterial({color:'#251b18',roughness:.23});
    this.iris=new THREE.MeshStandardMaterial({color:'#663b1e',roughness:.24});
    this.glint=new THREE.MeshBasicMaterial({color:'#fff8df'});
    this.pink=new THREE.MeshStandardMaterial({color:'#c87979',roughness:.65});
    this.collar=new THREE.MeshStandardMaterial({color:'#385f53',roughness:.7});
    this.brass=new THREE.MeshStandardMaterial({color:'#dbad58',metalness:.65,roughness:.35});
  }

  ellipsoid(parent, material, scale, position=[0,0,0]) {
    const mesh=new THREE.Mesh(Math.max(...scale)<.055?this.detailSphere:this.sphere,material);
    mesh.scale.set(...scale);mesh.position.set(...position);
    mesh.castShadow=mesh.receiveShadow=true;
    parent.add(mesh);
    return mesh;
  }

  curve(parent, points, radius, material) {
    const mesh=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p))),20,radius,8,false),material);
    mesh.castShadow=true;parent.add(mesh);return mesh;
  }

  add(geom) {
    const {name,body,type,size}=geom;
    if(body!=='torso' && !/^(fl|fr|rl|rr)_(hip|thigh|shin)$/.test(body??''))return null;
    const group=new THREE.Group();
    group.name='puppy:'+String(name??body);
    if(name==='torso_geom') {
      this.ellipsoid(group,this.gold,[.30,.153,.145]);
      this.ellipsoid(group,this.gold,[.12,.13,.14],[.20,0,.055]);
      this.ellipsoid(group,this.cream,[.21,.12,.075],[.025,0,-.067]);
      // A tapered, upward-curved tail, pivoting only as cosmetic expression.
      this.tail=new THREE.Group();this.tail.position.set(-.265,0,.035);group.add(this.tail);
      this.curve(this.tail,[[0,0,0],[-.075,0,.025],[-.15,0,.095],[-.18,0,.16]],.025,this.gold);
      this.ellipsoid(this.tail,this.cream,[.024,.025,.04],[-.18,0,.16]);
    } else if(name==='head') {
      this.ellipsoid(group,this.gold,[.126,.111,.12],[-.015,0,.03]);
      this.ellipsoid(group,this.cream,[.09,.079,.055],[.092,0,-.014]);
      for(const side of [-1,1]) {
        this.ellipsoid(group,this.cream,[.052,.037,.03],[.098,side*.038,-.008]);
        this.ellipsoid(group,this.honey,[.029,.023,.025],[.063,side*.078,.067]);
        const eye=new THREE.Group();eye.position.set(.080,side*.079,.069);group.add(eye);
        this.ellipsoid(eye,this.black,[.019,.017,.021]);
        this.ellipsoid(eye,this.iris,[.007,.013,.016],[.014,0,0]);
        this.ellipsoid(eye,this.black,[.007,.008,.012],[.019,0,0]);
        this.ellipsoid(eye,this.glint,[.003,.004,.004],[.023,-.004,.006]);
        this.eyes.push(eye);
        this.ellipsoid(group,this.gold,[.043,.025,.018],[.041,side*.077,.096]);
        const ear=new THREE.Group();ear.position.set(-.055,side*.091,.095);group.add(ear);
        this.ellipsoid(ear,this.honey,[.049,.035,.105],[-.015,side*.028,-.067]);
        this.ellipsoid(ear,this.gold,[.043,.032,.076],[-.021,side*.031,-.093]);
        this.ears.push({ear,side});
      }
      this.ellipsoid(group,this.black,[.026,.036,.023],[.170,0,.004]);
      this.ellipsoid(group,this.glint,[.003,.008,.002],[.193,-.009,.014]);
      this.curve(group,[[.163,-.050,-.035],[.174,-.028,-.043],[.179,0,-.043],[.174,.028,-.043],[.163,.05,-.035]],.0025,this.black);
      this.ellipsoid(group,this.pink,[.018,.014,.006],[.163,0,-.047]);
      const band=new THREE.Mesh(new THREE.TorusGeometry(.102,.012,8,40),this.collar);
      band.rotation.y=Math.PI/2;band.position.set(-.067,0,-.008);group.add(band);
      this.ellipsoid(group,this.brass,[.007,.021,.025],[-.053,-.089,-.072]);
    } else if(name?.endsWith('_foot')) {
      this.ellipsoid(group,this.gold,[.052,.041,.031],[.008,0,0]);
      for(const y of [-.024,0,.024])this.ellipsoid(group,this.cream,[.023,.014,.021],[.040,y,-.005]);
    } else if(type==='sphere') {
      this.ellipsoid(group,this.gold,[.055,.040,.093],[0,0,-.020]);
    } else if(type==='capsule') {
      const radius=body.endsWith('_thigh')?.045:.033;
      const leg=new THREE.Mesh(new THREE.CapsuleGeometry(radius,size[1]*2,4,12),this.gold);
      leg.rotation.x=Math.PI/2;
      leg.castShadow=leg.receiveShadow=true;group.add(leg);
    }
    // Batch static surface details by material; keep animated eyes/ears separate.
    const batches=new Map();
    for(const child of [...group.children])if(child.isMesh) {
      child.updateMatrix();
      const geometries=batches.get(child.material)??[];
      geometries.push(child.geometry.clone().applyMatrix4(child.matrix));
      batches.set(child.material,geometries);group.remove(child);
    }
    for(const [material,geometries] of batches) {
      const mesh=new THREE.Mesh(mergeGeometries(geometries),material);
      mesh.castShadow=mesh.receiveShadow=true;group.add(mesh);
      for(const geometry of geometries)geometry.dispose();
    }
    this.parts.set(geom.id,group);
    return group;
  }

  animate(time,speed,blink) {
    // No torso bobbing or fabricated leg cycles. Ears/tail are non-colliding art.
    const excitement=Math.min(1,Math.max(0,speed)/.35);
    if(this.tail)this.tail.rotation.x=Math.sin(time*7)*(.14+excitement*.28);
    for(const {ear,side} of this.ears)ear.rotation.x=side*(.12+Math.sin(time*12)*.035*excitement);
    for(const eye of this.eyes)eye.scale.z=blink;
  }

  stats() {
    const geometries=new Set();
    let triangles=0,draws=0;
    for(const part of this.parts.values())part.traverse(object=>{
      if(!object.isMesh)return;
      draws++;
      geometries.add(object.geometry);
      triangles+=(object.geometry.index?.count??object.geometry.attributes.position.count)/3;
    });
    let geometryBytes=0;
    for(const geometry of geometries) {
      for(const attribute of Object.values(geometry.attributes))geometryBytes+=attribute.array.byteLength;
      geometryBytes+=geometry.index?.array.byteLength??0;
    }
    return {physicalAttachments:this.parts.size,triangles,draws,geometryBytes,texturePixels:256*256};
  }
}
