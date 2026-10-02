export const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const fmt = value => (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3);
export const preview = vector => `[${(vector.length > 4 ? [...vector.slice(0, 2).map(fmt), '…', fmt(vector.at(-1))] : vector.map(fmt)).join(', ')}]`;
export function vector(label, values, kind = '') {
  return `<div class="vector-label">${label} <small>(${values.length} × 1)</small></div><div class="vector ${kind}" aria-label="Column vector">${values.map(v => `│ ${fmt(v).padStart(7)} │`).join('\n')}</div>`;
}
export function matrix(label, rows) {
  return `<h3>${label}</h3><div class="matrix-wrap"><table><tbody>${rows.map(row => `<tr>${row.map(v => `<td>${fmt(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
export function renderSequence(model, selected) {
  const { sequence, embeddingMap } = model;
  document.querySelector('#counts').textContent = `${sequence.original.length} TOKENS / ${sequence.maximum} POSITIONS`;
  document.querySelector('#tokens').innerHTML = sequence.tokens.map((token, i) => `<button class="token ${token === '<PAD>' ? 'pad' : ''} ${i === selected ? 'active' : ''}" data-step="${i}" aria-label="Inspect timestep ${i + 1}: ${escape(token)}" aria-pressed="${i === selected}"><small>t${i + 1}</small>${escape(token)}</button>`).join('');
  const result = sequence.padCount ? `${sequence.padCount} PAD tokens were added.` : sequence.truncated.length ? `the final ${sequence.truncated.length} tokens were truncated.` : 'no padding or truncation was needed.';
  document.querySelector('#sequence-note').textContent = `The sentence contains ${sequence.original.length} tokens. Maximum sequence length is ${sequence.maximum}, so ${result}`;
  document.querySelector('#truncated').innerHTML = sequence.truncated.length ? `<span class="hint">Truncated: </span>${sequence.truncated.map(t => `<span class="truncated">${escape(t)}</span>`).join('')}` : '';
  document.querySelector('#embedding-table').innerHTML = `<table><thead><tr><th>Token</th>${Array.from({length:model.E},(_,i)=>`<th>x[${i+1}]</th>`).join('')}</tr></thead><tbody>${[...embeddingMap].map(([token, values]) => `<tr><th>${escape(token)}</th>${values.map(v=>`<td>${fmt(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
