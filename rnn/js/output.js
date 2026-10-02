import { random } from './random.js';
import { matvec } from './math.js';
import { vector, matrix, fmt, escape } from './render.js';

export function softmax(scores) {
  const maximum = Math.max(...scores);
  const exps = scores.map(v => Math.exp(v - maximum));
  const total = exps.reduce((a,b) => a+b,0);
  return exps.map(v => v/total);
}
export function outputLayer(steps, H, O, seed, initialization, mode) {
  const rng = random(seed, 'output-parameters');
  const bound = initialization === 'xavier' ? Math.sqrt(6/(H+O)) : 0.5/Math.sqrt(H);
  const Wy = Array.from({length:O},()=>Array.from({length:H},()=>initialization==='zeros'?0:(rng()*2-1)*bound));
  const by = Array(O).fill(0);
  const last = steps.findLastIndex(s=>s.token!=='<PAD>');
  const results = steps.map((s,i)=> {
    if(mode==='none' || s.token==='<PAD>' || (mode==='final' && i!==last)) return null;
    const scores=matvec(Wy,s.h).map((v,j)=>v+by[j]);
    const probabilities=softmax(scores);
    return {scores,probabilities,prediction:probabilities.indexOf(Math.max(...probabilities))};
  });
  return {Wy,by,results};
}
export const classLabel = (model,i) => model.outputMode==='every' && model.O===3 ? ['Noun','Verb','Other'][i] : `Class ${i+1}`;
export function renderOutput(model, selected) {
  const host=document.querySelector('#output-calculation');
  host.hidden=model.outputMode==='none';
  if(host.hidden) { host.innerHTML=''; return; }
  const result=model.output.results[selected];
  host.innerHTML=`<h2>From memory to prediction</h2><p class="badge">UNTRAINED DEMONSTRATION</p><p>Hidden size H = <strong>${model.H}</strong> is memory size. Output size O = <strong>${model.O}</strong> is the number of classes. Shared output parameters: Wᵧ, bᵧ.</p><p class="hint">These random weights have not learned meaningful labels. Predictions use only the current and preceding tokens. Predictions do not feed into the next RNN cell.</p>`;
  if(!result) {
    host.innerHTML+=`<p class="operation-intro">${selected<0?'Select Next Step to begin.':model.steps[selected].token==='<PAD>'?'Padded position—output ignored.':'The output is taken only from the final real token.'}</p>`;
    return;
  }
  host.innerHTML+=`<div class="hero-equation">z<sub>${selected+1}</sub><sup>out</sup> = W<sub>y</sub>h<sub>${selected+1}</sub> + b<sub>y</sub><br>y<sub>${selected+1}</sub> = softmax(z<sub>${selected+1}</sub><sup>out</sup>)</div>
    <p>Wᵧ: ${model.O} × ${model.H} · h: ${model.H} × 1 · bᵧ, z<sup>out</sup>, y: ${model.O} × 1</p>
    <div class="output-vectors">${vector('Hidden state h',model.steps[selected].h)}${vector('Output scores zᵒᵘᵗ',result.scores)}${vector('Probabilities y',result.probabilities)}</div>
    <div class="scroll"><table><thead><tr><th>Illustrative label</th><th>Probability</th></tr></thead><tbody>${result.probabilities.map((v,i)=>`<tr><th>${escape(classLabel(model,i))}${i===result.prediction?' (selected)':''}</th><td>${(v*100).toFixed(2)}%</td></tr>`).join('')}</tbody></table></div>
    <p class="hint">Largest probability selects the label. Ties select the first class. Softmax subtracts the largest score before exponentiation for numerical stability.</p>
    <label for="output-class">Inspect output class</label><select id="output-class">${result.scores.map((_,i)=>`<option value="${i}">${escape(classLabel(model,i))}</option>`).join('')}</select><div id="output-arithmetic"></div>
    <details><summary>Inspect shared output weights and bias</summary>${matrix('Wᵧ',model.output.Wy)}${vector('bᵧ',model.output.by)}</details>`;
  const arithmetic=()=>{
    const j=Number(document.querySelector('#output-class').value), h=model.steps[selected].h;
    document.querySelector('#output-arithmetic').innerHTML=`<p class="unit-equation">zᵒᵘᵗ[${j+1}] = ${model.output.Wy[j].map((w,k)=>`(${fmt(w)} × ${fmt(h[k])})`).join(' + ')} + ${fmt(model.output.by[j])} ≈ ${fmt(result.scores[j])}<br>y[${j+1}] = exp(${fmt(result.scores[j])} − m) / Σ exp(zᵒᵘᵗ[k] − m) ≈ ${fmt(result.probabilities[j])}<br>m = max(zᵒᵘᵗ) ≈ ${fmt(Math.max(...result.scores))}</p>`;
  };
  document.querySelector('#output-class').addEventListener('change',arithmetic);arithmetic();
}
