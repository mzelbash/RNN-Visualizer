import { preprocess } from './sequence.js';
import { embeddings } from './embeddings.js';
import { initialize, forward } from './math.js';
import { renderSequence } from './render.js';
import { renderNetwork, effectiveView } from './visualization.js';
import { renderDetail } from './details.js';
import { renderOperation } from './operation.js';
import { outputLayer, renderOutput } from './output.js';
const $ = id => document.getElementById(id);
// -1 is the ready state: no token has been processed or revealed yet.
let model, selected = -1, timer = null, shared = false;
function stop() { clearInterval(timer); timer = null; $('run').textContent = 'Run All'; }
function render(open = false, scroll = false) {
  const mode = effectiveView(model.E, model.H, $('view').value);
  $('view').value = mode; $('view').options[0].disabled = model.E > 8 || model.H > 8;
  $('mode-note').textContent = `Hidden size = ${model.H}. The RNN produces ${model.H} hidden values at every timestep.` + (model.E>8||model.H>8?' Vector View is used automatically for dimensions above 8.':' Each circle represents one value.');
  renderSequence(model, selected); renderNetwork(model, selected, mode, shared); renderDetail(model, selected, open);
  renderOperation(model, selected);
  renderOutput(model, selected);
  if(model.outputMode!=='none') $('mode-note').textContent += ` Output size = ${model.O}. Untrained demonstration; shared output weights Wᵧ and bᵧ. Click an output to inspect.`;
  $('previous').disabled = selected === -1; $('next').disabled = selected === model.steps.length-1;
  $('progress').textContent = selected === -1 ? `Start · 0 of ${model.steps.length} processed` : `Step ${selected+1} of ${model.steps.length}`;
  if(selected === -1) $('network').scrollLeft = 0;
  if(scroll && selected >= 0) {
    const viewport=$('network'), cell=viewport.querySelector(`[data-step="${selected}"]`);
    const box=cell.getBoundingClientRect(), parent=viewport.getBoundingClientRect();
    if(box.right>parent.right || box.left<parent.left) viewport.scrollLeft += box.left-parent.left-35;
  }
}
function choose(index, open=true) { stop(); selected = Math.max(-1,Math.min(model.steps.length-1,index)); render(open,true); }
function build(event) {
  event?.preventDefault(); if(!$('controls').reportValidity()) return; stop();
  const E=Number($('embedding').value), H=Number($('hidden').value), seed=Number($('seed').value), initial=$('initial').value;
  const sequence=preprocess($('sentence').value,Number($('maximum').value));
  const embeddingMap=embeddings(sequence.tokens,E,seed), parameters=initialize(E,H,seed,initial,$('weights').value);
  model={E,H,seed,initial,sequence,embeddingMap,parameters,steps:forward(sequence.tokens,embeddingMap,parameters)};
  model.outputMode=$('output-mode').value; model.O=Number($('output-size').value);
  model.output=outputLayer(model.steps,H,model.O,seed,$('weights').value,model.outputMode);
  selected=-1; $('dirty').textContent=`Built with ${$('weights').selectedOptions[0].text}, seed ${seed}. Bias is zero.`; render(); $('network').scrollLeft=0;
}
$('controls').addEventListener('submit',build);
$('controls').addEventListener('input',()=>{$('dirty').textContent='Settings changed. Select Build RNN to apply.';stop();});
$('reset').addEventListener('click',()=>choose(-1,false));
$('previous').addEventListener('click',()=>choose(selected-1));
$('next').addEventListener('click',()=>choose(selected+1));
for(const id of ['tokens','network']) {
  $(id).addEventListener('click',event=>{ const target=event.target.closest('[data-step]'); if(target) {choose(Number(target.dataset.step)); if(target.hasAttribute('data-output')) $('output-calculation').scrollIntoView({block:'start'});} });
}
$('network').addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){const target=event.target.closest('[data-step]');if(target){event.preventDefault();const output=target.hasAttribute('data-output');choose(Number(target.dataset.step));$('network').querySelector(`[data-step="${selected}"]${output?'[data-output]':':not([data-output])'}`).focus();if(output)$('output-calculation').scrollIntoView({block:'start'});}}});
$('view').addEventListener('change',()=>render($('calculation')?.open));
$('shared').addEventListener('click',()=>{shared=!shared;$('shared').setAttribute('aria-pressed',String(shared));$('shared-note').hidden=!shared;render($('calculation')?.open);});
$('run').addEventListener('click',()=>{
  if(timer){stop();return;} selected=-1;render(false,true);$('run').textContent='Pause';
  timer=setInterval(()=>{selected++;render(true,true);if(selected===model.steps.length-1)stop();},1500);
});
const operationPanel = document.createElement('section');
operationPanel.className = 'panel operation-panel';
operationPanel.setAttribute('aria-labelledby','operation-heading');
operationPanel.innerHTML = '<div class="section-head"><h2 id="operation-heading">Inside this timestep</h2><span id="operation-step" class="badge"></span></div><div id="operation"></div>';
document.querySelector('.workspace').append(operationPanel);
const outputPanel=document.createElement('section');
outputPanel.id='output-calculation'; outputPanel.className='panel operation-panel';
document.querySelector('.workspace').append(outputPanel);
for (const [button, panel, label, className] of [
  ['toggle-controls','controls-panel','controls','controls-hidden'],
  ['toggle-details','details-panel','details','details-hidden']
]) {
  $(button).addEventListener('click', () => {
    const hidden = !$(panel).hidden;
    $(panel).hidden = hidden;
    document.querySelector('main').classList.toggle(className, hidden);
    $(button).setAttribute('aria-expanded', String(!hidden));
    $(button).textContent = `${hidden ? 'Show' : 'Hide'} ${label}`;
  });
}
build();
