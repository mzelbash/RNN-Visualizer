import { vector, matrix, escape, fmt } from './render.js';
export function renderDetail(model, index, open) {
  const s = model.steps[index], { Wx, Wh, b } = model.parameters;
  const container = document.querySelector('#detail');
  if (window.MathJax?.typesetClear) window.MathJax.typesetClear([container]);
  if (index < 0) {
    container.innerHTML = '<p class="step-label">START</p><h3>Ready for the first token</h3><p>No token has been processed yet. Select Next Step to begin.</p>' + vector('Initial hidden state h<sub>0</sub>', model.parameters.h0, 'previous');
    return;
  }
  container.innerHTML = `<p class="step-label">TIMESTEP ${s.t} / ${model.steps.length}</p><h3 class="token-title">Token: ${escape(s.token)}</h3>`+
    vector(`Current input x<sub>${s.t}</sub>`, s.x, 'input') + vector(`Previous hidden state h<sub>${s.t-1}</sub>`, s.previous, 'previous') + vector(`New hidden state h<sub>${s.t}</sub>`, s.h) +
    `<h3>Dimensions</h3><div class="dimension-list"><span>x${s.t}: ${model.E} × 1</span><span>h${s.t-1}: ${model.H} × 1</span><span>Wₓ: ${model.H} × ${model.E}</span><span>Wₕ: ${model.H} × ${model.H}</span><span>b: ${model.H} × 1</span><span>h${s.t}: ${model.H} × 1</span></div>
    <details id="calculation" ${open?'open':''}><summary>Show Calculation</summary><div id="equation" class="formula">h<sub>${s.t}</sub> = tanh(W<sub>x</sub>x<sub>${s.t}</sub> + W<sub>h</sub>h<sub>${s.t-1}</sub> + b)</div><p class="hint">Column vectors. Display rounded to 3 decimals; calculations use full precision.</p>
    ${vector('Wₓx'+s.t,s.inputTerm,'input')}${vector('Wₕh'+(s.t-1),s.recurrentTerm,'previous')}${vector('Bias b',b)}${vector('z = Wₓx + Wₕh + b',s.z)}${vector('h'+s.t+' = tanh(z)',s.h)}
    <label for="unit">Inspect a hidden unit</label><select id="unit">${s.h.map((_,i)=>`<option value="${i}">Hidden unit ${i+1}</option>`).join('')}</select><div id="unit-calculation"></div>
    <details id="parameters"><summary>Inspect shared Wₓ, Wₕ, b</summary><div id="parameter-values"></div></details></details>`;
  const renderUnit = () => {
    const j = Number(document.querySelector('#unit').value);
    const terms = (weights, values) => weights.map((w,k)=>`(${fmt(w)} × ${fmt(values[k])})`).join(' + ');
    document.querySelector('#unit-calculation').innerHTML = `<p class="unit-equation">Wₓ row ${j+1} · x = ${terms(Wx[j],s.x)} ≈ ${fmt(s.inputTerm[j])}<br><br>Wₕ row ${j+1} · h = ${terms(Wh[j],s.previous)} ≈ ${fmt(s.recurrentTerm[j])}<br><br>z[${j+1}] ≈ ${fmt(s.inputTerm[j])} + ${fmt(s.recurrentTerm[j])} + ${fmt(b[j])} = ${fmt(s.z[j])}<br>h[${j+1}] = tanh(z[${j+1}]) ≈ ${fmt(s.h[j])}</p>`;
  };
  document.querySelector('#unit').addEventListener('change',renderUnit); renderUnit();
  document.querySelector('#parameters').addEventListener('toggle',event => {
    if(event.target.open && !document.querySelector('#parameter-values').hasChildNodes()) document.querySelector('#parameter-values').innerHTML=matrix(`Wₓ (${model.H} × ${model.E})`,Wx)+matrix(`Wₕ (${model.H} × ${model.H})`,Wh)+vector('b',b);
  });
  if (window.MathJax?.typesetPromise) {
    document.querySelector('#equation').textContent = `\\(h_{${s.t}}=\\tanh(W_x x_{${s.t}}+W_h h_{${s.t-1}}+b)\\)`;
    window.MathJax.typesetPromise([document.querySelector('#equation')]).catch(()=>{});
  }
}
