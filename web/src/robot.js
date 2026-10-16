import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// Original procedural research-robot artwork. Every assembly follows its actual
// MuJoCo geometry. The pickup is the existing vertical magnetic contact tool,
// not an independently simulated six-axis arm or an invented grasp animation.
export class RobotSkin {
  constructor() {
    this.parts = new Map();
    this.materials = {
      shell: new THREE.MeshStandardMaterial({color:'#e4e9df', roughness:.34, metalness:.42}),
      dark: new THREE.MeshStandardMaterial({color:'#1c2c2d', roughness:.45, metalness:.56}),
      rubber: new THREE.MeshStandardMaterial({color:'#182322', roughness:.96, metalness:0}),
      metal: new THREE.MeshStandardMaterial({color:'#8caca9', roughness:.29, metalness:.87}),
      copper: new THREE.MeshStandardMaterial({color:'#d1a26f', roughness:.36, metalness:.65}),
      lens: new THREE.MeshStandardMaterial({color:'#173e42', roughness:.1, metalness:.8}),
      light: new THREE.MeshStandardMaterial({color:'#a8edc5', emissive:'#64d6b2', emissiveIntensity:1.4, roughness:.23}),
    };
  }

  mesh(group, geometry, material, position=[0,0,0]) {
    const mesh = new THREE.Mesh(geometry, this.materials[material]);
    mesh.position.set(...position);mesh.castShadow=mesh.receiveShadow=true;group.add(mesh);return mesh;
  }

  box(group, size, material, position=[0,0,0], radius=.009) {
    return this.mesh(group,new RoundedBoxGeometry(...size,2,Math.min(radius,...size.map(n=>n*.2))),material,position);
  }

  cylinder(group, radius, length, material, position=[0,0,0], axis='z') {
    const mesh = this.mesh(group,new THREE.CylinderGeometry(radius,radius,length,16),material,position);
    if(axis==='z')mesh.rotation.x=Math.PI/2;
    if(axis==='x')mesh.rotation.z=Math.PI/2;
    return mesh;
  }

  add(geom) {
    const {name,body,type,size}=geom;
    if(body!=='torso' && body!=='gripper' && !/^(fl|fr|rl|rr)_(hip|thigh|shin)$/.test(body??''))return null;
    const group=new THREE.Group();group.name=`robot:${name??body}`;
    if(name==='torso_geom') {
      this.box(group,[.54,.265,.15],'dark');
      this.box(group,[.47,.257,.065],'shell',[.006,0,.067],.025);
      this.box(group,[.31,.20,.018],'metal',[-.035,0,.108]);
      for(const y of [-.105,.105]) {
        this.box(group,[.42,.018,.012],'copper',[-.02,y,.109],.002);
        this.box(group,[.32,.008,.008],'light',[-.045,y*1.25,.02],.002);
      }
      for(const x of [-.16,-.11,-.06,-.01,.04,.09])this.box(group,[.016,.125,.004],'dark',[x,0,.12],.001);
      // Stationary rail and mount. Its carriage is attached to gripper_geom.
      this.box(group,[.12,.13,.06],'dark',[.30,0,-.045]);
      for(const y of [-.043,.043])this.cylinder(group,.008,.31,'metal',[.428,y,-.16]);
      this.box(group,[.055,.12,.025],'copper',[.428,0,-.002]);
      this.box(group,[.055,.12,.025],'dark',[.428,0,-.308]);
      this.box(group,[.09,.16,.04],'dark',[-.225,0,.094]);
      for(const y of [-.06,.06])this.box(group,[.007,.02,.012],'light',[-.274,y,.09],.002);
    } else if(name==='head') {
      this.box(group,[.19,.208,.15],'shell',[0,0,0],.025);
      this.box(group,[.025,.184,.101],'dark',[.094,0,.007],.016);
      for(const y of [-.052,.052]) {
        this.cylinder(group,.033,.018,'metal',[.11,y,.012],'x');
        this.cylinder(group,.025,.021,'lens',[.12,y,.012],'x');
        this.cylinder(group,.010,.022,'light',[.122,y,.012],'x');
      }
      this.box(group,[.012,.051,.009],'light',[.111,0,-.045],.002);
      this.box(group,[.07,.15,.012],'dark',[-.015,0,.08]);
      for(const y of [-.08,.08])this.cylinder(group,.016,.007,'copper',[-.055,y,.084]);
    } else if(name==='gripper_geom') {
      this.box(group,[.083,.113,.041],'dark',[0,0,.03]);
      this.box(group,[.06,.09,.016],'copper',[0,0,.056]);
      this.cylinder(group,.055,.038,'metal',[0,0,-.006],'x');
      this.cylinder(group,.043,.042,'rubber',[.004,0,-.006],'x');
      this.box(group,[.011,.08,.009],'light',[.044,0,.034],.002);
      for(const y of [-.058,.058]) {
        this.box(group,[.085,.012,.055],'shell',[.025,y,-.005],.004);
        this.box(group,[.018,.022,.047],'rubber',[.06,y,-.008],.003);
      }
    } else if(name?.endsWith('_foot')) {
      this.mesh(group,new THREE.SphereGeometry(1,16,10),'rubber').scale.set(.044,.038,.025);
      this.box(group,[.047,.043,.021],'metal',[0,0,.018],.004);
    } else if(type==='sphere') {
      this.cylinder(group,.046,.072,'dark',[0,0,0],'x');
      this.cylinder(group,.032,.076,'copper',[0,0,0],'x');
    } else if(type==='capsule') {
      const radius=body.endsWith('_thigh')?.031:.022;
      const leg=this.mesh(group,new THREE.CapsuleGeometry(radius,size[1]*2,4,10),'shell');
      leg.rotation.x=Math.PI/2;
      this.box(group,[radius*.75,radius*.7,size[1]*1.5],'dark',[radius*.8,0,0],.002);
      for(const z of [-size[1],size[1]])this.cylinder(group,radius*1.13,.047,'dark',[0,0,z],'x');
    }
    // One draw per material/physical assembly, including the pickup carriage.
    const batches=new Map();
    for(const child of [...group.children])if(child.isMesh) {
      child.updateMatrix();
      const geometries=batches.get(child.material)??[];
      const transformed=child.geometry.clone().applyMatrix4(child.matrix);
      geometries.push(transformed.index?transformed.toNonIndexed():transformed);
      if(transformed.index)transformed.dispose();
      batches.set(child.material,geometries);group.remove(child);
      child.geometry.dispose();
    }
    for(const [material,geometries] of batches) {
      const mesh=new THREE.Mesh(mergeGeometries(geometries),material);
      mesh.castShadow=mesh.receiveShadow=true;group.add(mesh);
      for(const geometry of geometries)geometry.dispose();
    }
    this.parts.set(geom.id,group);return group;
  }

  stats() {
    let triangles=0,draws=0,geometryBytes=0;
    const geometries=new Set();
    for(const group of this.parts.values())group.traverse(mesh=>{
      if(!mesh.isMesh)return;draws++;geometries.add(mesh.geometry);
      triangles+=(mesh.geometry.index?.count??mesh.geometry.attributes.position.count)/3;
    });
    for(const geometry of geometries) {
      for(const attribute of Object.values(geometry.attributes))geometryBytes+=attribute.array.byteLength;
      geometryBytes+=geometry.index?.array.byteLength??0;
    }
    return {physicalAttachments:this.parts.size,triangles,draws,geometryBytes,externalAssets:0,pickup:'physical vertical magnetic contact tool'};
  }
}
