export function pcmWave(chunks,sourceRate,targetRate=16000) {
  const length=chunks.reduce((sum,chunk)=>sum+chunk.length,0);
  const source=new Float32Array(length);
  let at=0;
  for(const chunk of chunks){source.set(chunk,at);at+=chunk.length;}
  const samples=Math.floor(length*targetRate/sourceRate);
  const buffer=new ArrayBuffer(44+samples*2);
  const view=new DataView(buffer);
  const text=(offset,value)=>{for(let i=0;i<value.length;i++)view.setUint8(offset+i,value.charCodeAt(i));};
  text(0,'RIFF');view.setUint32(4,36+samples*2,true);text(8,'WAVE');text(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
  view.setUint32(24,targetRate,true);view.setUint32(28,targetRate*2,true);
  view.setUint16(32,2,true);view.setUint16(34,16,true);text(36,'data');view.setUint32(40,samples*2,true);
  for(let i=0;i<samples;i++) {
    const start=Math.floor(i*sourceRate/targetRate),end=Math.max(start+1,Math.floor((i+1)*sourceRate/targetRate));
    let value=0;
    for(let j=start;j<end&&j<length;j++)value+=source[j];
    value=Math.max(-1,Math.min(1,value/(end-start)));
    view.setInt16(44+i*2,Math.round(value*(value<0?32768:32767)),true);
  }
  return buffer;
}

export function setupVoice(button,input,notify,transcribeUrl='/api/transcribe',request=fetch) {
  let recording,timeout;
  const stop=async()=>{
    if(!recording)return;
    clearTimeout(timeout);
    const {context,stream,node,source,silent,chunks}=recording;
    recording=null;
    source.disconnect();node.disconnect();silent.disconnect();
    stream.getTracks().forEach(track=>track.stop());
    const sampleRate=context.sampleRate;
    await context.close();
    button.disabled=true;button.textContent='Transcribing…';button.setAttribute('aria-pressed','false');
    try {
      const response=await request(transcribeUrl,{method:'POST',headers:{'Content-Type':'audio/wav'},body:pcmWave(chunks,sampleRate)});
      const result=await response.json();
      if(!response.ok)throw new Error(result.detail??'Could not transcribe');
      input.value=result.text;input.focus();notify(`Heard “${result.text}”. Press Send to give the command.`);
    }catch(error){notify(error.message);}
    finally{button.disabled=false;button.textContent='● Speak';}
  };
  button.onclick=async()=>{
    if(recording){await stop();return;}
    let stream,context;
    try {
      if(!navigator.mediaDevices?.getUserMedia)throw new Error('Microphone capture needs HTTPS or a local browser connection. You can type any command.');
      stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false});
      context=new AudioContext();
      await context.audioWorklet.addModule(new URL('./pcm-worklet.js',import.meta.url));
      await context.resume();
      const source=context.createMediaStreamSource(stream);
      const node=new AudioWorkletNode(context,'command-recorder');
      const silent=context.createGain();silent.gain.value=0;
      const chunks=[];node.port.onmessage=({data})=>chunks.push(data);
      source.connect(node);node.connect(silent);silent.connect(context.destination);
      recording={stream,context,source,node,silent,chunks};
      button.textContent='■ Finish';button.setAttribute('aria-pressed','true');
      notify('Listening for a short command. Finish when ready; audio stays on this server.');
      timeout=setTimeout(stop,6000);
    }catch(error){stream?.getTracks().forEach(track=>track.stop());await context?.close();notify(error.message);}
  };
  window.addEventListener('pagehide',()=>{
    clearTimeout(timeout);recording?.stream.getTracks().forEach(track=>track.stop());
    recording?.context.close();recording=null;
  });
}
