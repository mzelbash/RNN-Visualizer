import { escape, fmt, preview } from './render.js';
import { classLabel } from './output.js';
export const effectiveView = (E, H, requested) => E > 8 || H > 8 ? 'vector' : requested;
export function renderNetwork(model, selected, mode, shared) {
  const stride = 190, start = 90, width = start + model.steps.length * stride;
  const height = model.H <= 8 ? 310 + model.H * 20 : 330;
  const hasOutput=model.outputMode && model.outputMode!=='none';
  const text = (x,y,value,extra='') => `<text x="${x}" y="${y}" text-anchor="middle" ${extra}>${escape(value)}</text>`;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height+(hasOutput?150:0)}" viewBox="0 ${hasOutput?-30:120} ${width} ${height+(hasOutput?150:0)}" role="group" aria-label="${model.steps.length} RNN timesteps"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="context-stroke"/></marker></defs>`;
  if(hasOutput) svg+=text(width/2,-10,'UNTRAINED DEMONSTRATION · Shared output weights Wᵧ, bᵧ','font-size="14" font-weight="700" fill="#a34b72"');
  svg += text(30,179,'h₀','font-size="20" font-weight="700" fill="#b05c13"') + text(30,202,model.initial === 'zeros' ? 'zeros' : 'random','font-size="12" fill="#65758b"');
  model.steps.forEach((step, i) => {
    const x = start + i * stride, active = selected === i, seen = i <= selected;
    const color = active || shared ? '#7050c5' : '#aab5c8';
    svg += `<g><title>Hidden state carried from the previous timestep</title><path d="M${x - 50} 182 H${x - 5}" stroke="${active ? '#b05c13' : '#aab5c8'}" stroke-width="${active ? 3 : 2}" marker-end="url(#arrow)"/>`;
    if (i > 0) svg += text(x-27,166,`h${i}`,'font-size="15" font-weight="700" fill="#b05c13"');
    svg += '</g>';
    if(hasOutput) {
      const output=model.output.results[i];
      if(output) {
        svg+=`<g role="button" tabindex="0" data-step="${i}" data-output="true" aria-label="Inspect output at timestep ${i+1}"><title>Inspect shared output weights, scores, and probabilities</title><path d="M${x+70} 148 V106" stroke="#a34b72" stroke-width="2" marker-end="url(#arrow)"/><rect x="${x}" y="20" width="140" height="84" rx="10" fill="#fff0f6" stroke="${shared?'#7050c5':'#c68aa5'}" stroke-width="${shared?3:1.5}"/>`;
        svg+=text(x+70,43,`y${i+1} [${model.O}]`,'font-size="17" font-weight="700" fill="#a34b72"');
        svg+=text(x+70,67,seen?classLabel(model,output.prediction):'Not yet revealed','font-size="14" fill="#a34b72"');
        svg+=text(x+70,90,seen?`${(output.probabilities[output.prediction]*100).toFixed(1)}% · inspect`:'Output layer','font-size="12" fill="#a34b72"')+'</g>';
      } else if(step.token==='<PAD>') svg+=text(x+70,74,'PAD · output ignored','font-size="12" fill="#78879a"');
    }
    svg += `<g role="button" tabindex="0" data-step="${i}" aria-label="Inspect timestep ${i+1}: ${escape(step.token)}" aria-pressed="${active}"><title>Uses the current input and previous hidden state to compute the new hidden state.</title>`;
    svg += `<rect x="${x-3}" y="128" width="146" height="${height-16}" rx="12" fill="${active ? '#f3efff' : '#fff'}" stroke="${active ? '#c8b9ed' : 'none'}"/>`;
    svg += `<rect x="${x}" y="150" width="140" height="64" rx="12" fill="${active||shared?'#e8defd':'#f0f2f8'}" stroke="${color}" stroke-width="${active||shared?3:2}" ${step.token==='<PAD>'?'stroke-dasharray="5 3"':''}/>`;
    if(mode==='neuron') step.h.forEach((v,j)=> {svg+=`<circle cx="${x+70+(j-(model.H-1)/2)*15}" cy="182" r="6" fill="${seen?'#8b69cf':'#fff'}" stroke="#7050c5"><title>Hidden unit ${j+1}: ${seen?fmt(v):'not revealed yet'}</title></circle>`;});
    else svg += text(x+70,189,`${model.H} hidden units`,'font-size="16" font-weight="700" fill="#7050c5"');
    // Inputs enter from below; the recurrent state travels left to right.
    svg += `<path d="M${x+70} 257 V218" stroke="#2366c6" stroke-width="2.5" marker-end="url(#arrow)"/>`;
    svg += `<g><title>Current token embedding: ${escape(step.token)}</title><rect x="${x+12}" y="260" width="116" height="${mode==='neuron'?52:38}" rx="7" fill="${active ? '#e2efff' : '#f0f5fd'}" stroke="#a4c4ed" ${step.token==='<PAD>'?'stroke-dasharray="4 3"':''}/>`;
    svg += text(x+70,283,`x${i+1} [${model.E}]`,'font-size="18" font-weight="700" fill="#2366c6"');
    if(mode==='neuron') step.x.forEach((v,j)=> {svg+=`<circle cx="${x+70+(j-(model.E-1)/2)*12}" cy="300" r="4" fill="#4d8ddd"><title>x[${j+1}] = ${fmt(v)}</title></circle>`;});
    const short = step.token.length > 13 ? step.token.slice(0,10)+'…' : step.token;
    svg += '</g>'+text(x+70,340,short,`font-size="22" font-weight="800" fill="${step.token==='<PAD>'?'#78879a':'#2366c6'}"`);
    svg += text(x+70,368,`TIMESTEP ${i+1}`,'font-size="13" font-weight="700" fill="#65758b" letter-spacing="1"');
    // An inspection readout, not an output-layer branch. Future outputs can sit above cells.
    if (active) {
      svg += text(x+70,398,`Hidden state h${i+1}`,'font-size="15" font-weight="700" fill="#087c69"');
      if (model.H <= 8) {
        const bottom = 412 + model.H * 20;
        svg += `<path d="M${x+38} 408 H${x+32} V${bottom} H${x+38} M${x+102} 408 H${x+108} V${bottom} H${x+102}" fill="none" stroke="#087c69" stroke-width="2"/>`;
        step.h.forEach((value,j) => { svg += text(x+70,426+j*20,fmt(value),'font-size="16" font-weight="700" fill="#087c69"'); });
      } else {
        svg += text(x+70,422,preview(step.h),'font-size="11" font-weight="700" fill="#087c69"');
        svg += text(x+70,440,`${model.H} values`,'font-size="12" fill="#476b68"');
      }
    }
    svg += '</g>';
    if(hasOutput && model.output.results[i]) svg += `<path d="M${x+70} 148 V106" stroke="#a34b72" stroke-width="2" marker-end="url(#arrow)"/>`;
  });
  document.querySelector('#network').innerHTML = svg + '</svg>';
}

