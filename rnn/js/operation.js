import { escape, fmt } from './render.js';
export function renderOperation(model, selected) {
  if (selected < 0) {
    document.querySelector('#operation-step').textContent = 'START';
    document.querySelector('#operation').innerHTML = '<p class="operation-intro"><strong>Ready to process the sentence.</strong> No timesteps have been processed yet.</p><p class="hero-equation">Select <span class="input-color">Next Step</span> to process the first token.</p><p class="hint">The first cell will combine x₁ with the initial hidden state h₀. Its calculation and result will appear here.</p>';
    return;
  }
  const s = model.steps[selected];
  const column = values => `<span class="math-column">${values.map(v => `<span>${fmt(v)}</span>`).join('')}</span>`;
  const term = (label, values, kind) => `<div class="math-term ${kind}"><div>${label}</div>${column(values)}</div>`;
  document.querySelector('#operation-step').textContent = `STEP ${s.t} · ${s.token}`;
  document.querySelector('#operation').innerHTML = `
    <p class="operation-intro">Read <strong>${escape(s.token)}</strong>. Combine the current input with the previous hidden state.</p>
    <div class="hero-equation"><span class="output-color">h<sub>${s.t}</sub></span> = tanh(<span class="input-color">W<sub>x</sub>x<sub>${s.t}</sub></span> + <span class="previous-color">W<sub>h</sub>h<sub>${s.t-1}</sub></span> + <span class="bias-color">b</span>)</div>
    <div class="operation-scroll"><div class="numeric-operation">
      ${term(`Wₓx${s.t}`,s.inputTerm,'input-color')}<span class="math-sign">+</span>
      ${term(`Wₕh${s.t-1}`,s.recurrentTerm,'previous-color')}<span class="math-sign">+</span>
      ${term('b',model.parameters.b,'bias-color')}<span class="math-sign">=</span>
      ${term('z',s.z,'sum-color')}<span class="math-sign activation-step">tanh<br>→</span>
      ${term(`h${s.t}`,s.h,'output-color')}
    </div></div><p class="hint">Each column contains ${model.H} values. Apply tanh to every entry of z. Values shown to 3 decimals; the calculation uses full precision.</p>`;
}
