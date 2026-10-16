// Actual Chromium frame capture via CDP and the installed system FFmpeg.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

export async function recordPage(context,page,out) {
  fs.mkdirSync(out,{recursive:true});
  const cdp=await context.newCDPSession(page),frames=[];
  cdp.on('Page.screencastFrame',async frame=>{
    const file=`frame-${String(frames.length).padStart(6,'0')}.jpg`;
    fs.writeFileSync(path.join(out,file),Buffer.from(frame.data,'base64'));
    frames.push({file,time:frame.metadata.timestamp});
    await cdp.send('Page.screencastFrameAck',{sessionId:frame.sessionId}).catch(()=>{});
  });
  await cdp.send('Page.startScreencast',{format:'jpeg',quality:85,maxWidth:1440,maxHeight:1440});
  return async()=>{
    await cdp.send('Page.stopScreencast');
    fs.writeFileSync(path.join(out,'frames.json'),JSON.stringify(frames,null,2));
    if(frames.length<2)throw new Error('Insufficient actual browser frames');
    const concat=frames.map((f,i)=>`file '${f.file}'\nduration ${Math.max(.001,(frames[i+1]?.time??f.time+.04)-f.time)}`).join('\n');
    fs.writeFileSync(path.join(out,'frames.ffconcat'),concat+'\n');
    execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',path.join(out,'frames.ffconcat'),'-c:v','libvpx-vp9','-crf','36','-b:v','0',path.join(out,'fetch.webm')]);
  };
}
