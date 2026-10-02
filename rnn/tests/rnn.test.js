import test from 'node:test';
import assert from 'node:assert/strict';
import { preprocess } from '../js/sequence.js';
import { embeddings } from '../js/embeddings.js';
import { initialize, forward } from '../js/math.js';
import { effectiveView } from '../js/visualization.js';
import { softmax, outputLayer } from '../js/output.js';
function build(sentence='The cat sat on the mat',max=8,E=4,H=3,seed=42,initial='zeros') {
  const sequence=preprocess(sentence,max), map=embeddings(sequence.tokens,E,seed), parameters=initialize(E,H,seed,initial);
  return {sequence,map,parameters,steps:forward(sequence.tokens,map,parameters)};
}
test('Output head is reproducible, dimensionally correct, and ignores PAD',()=>{
  const m=build(); const before=structuredClone(m.steps);
  const out=outputLayer(m.steps,3,4,42,'xavier','every');
  assert.deepEqual(out,outputLayer(m.steps,3,4,42,'xavier','every'));
  assert.equal(out.Wy.length,4);assert.ok(out.Wy.every(r=>r.length===3));
  assert.equal(out.results.filter(Boolean).length,6);
  assert.ok(out.results.filter(Boolean).every(r=>Math.abs(r.probabilities.reduce((a,b)=>a+b,0)-1)<1e-12));
  const scores=out.Wy.map(row=>row[0]*m.steps[0].h[0]+row[1]*m.steps[0].h[1]+row[2]*m.steps[0].h[2]);
  assert.deepEqual(out.results[0].scores,scores);assert.deepEqual(m.steps,before);
  const final=outputLayer(m.steps,3,4,42,'xavier','final');
  assert.deepEqual(final.results.map((r,i)=>r?i:null).filter(v=>v!==null),[5]);
  assert.deepEqual(final.results[5],out.results[5]);
  assert.ok(outputLayer(build('').steps,3,3,42,'small','final').results.every(r=>r===null));
  assert.ok(outputLayer(m.steps,3,3,42,'small','none').results.every(r=>r===null));
});
test('Stable softmax and zero output weights',()=>{
  const p=softmax([1000,1001]);assert.ok(Math.abs(p[0]-1/(1+Math.E))<1e-12);
  assert.deepEqual(softmax([-1000,-1000]),[0.5,0.5]);
  const out=outputLayer(build().steps,3,3,42,'zeros','every');
  assert.deepEqual(out.results[0].probabilities,[1/3,1/3,1/3]);assert.equal(out.results[0].prediction,0);
});
test('Case 1: six real tokens, two zero PAD vectors and eight real recurrent steps',()=>{
  const m=build(); assert.equal(m.sequence.original.length,6);assert.equal(m.sequence.padCount,2);assert.equal(m.steps.length,8);
  assert.deepEqual(m.map.get('<PAD>'),[0,0,0,0]);assert.ok(m.steps.every(s=>s.h.length===3));
  assert.deepEqual(m.steps[7].previous,m.steps[6].h);assert.notDeepEqual(m.steps[7].h,m.steps[6].h);
});
test('Case 2: truncation retains only maximum sequence length',()=>{
  const m=build('one two three four five six seven eight nine ten');
  assert.deepEqual(m.sequence.truncated,['nine','ten']);assert.equal(m.steps.length,8);
});
test('Case 3: repeated words share an embedding, case is preserved',()=>{
  const m=build('the cat and the dog');assert.strictEqual(m.steps[0].x,m.steps[3].x);
  assert.deepEqual(m.map.get('the'),build('the').map.get('the'));
  assert.notDeepEqual(embeddings(['The','the'],4,42).get('The'),m.map.get('the'));
});
test('Case 4: large dimensions force vector view',()=>{
  const m=build('test',12,128,128);assert.equal(m.steps[0].h.length,128);assert.equal(effectiveView(4,128,'neuron'),'vector');
  assert.equal(effectiveView(128,3,'neuron'),'vector');assert.ok(m.steps.every(s=>s.h.every(Number.isFinite)));
});
test('Case 5: column-vector dimensions',()=>{
  const {parameters:p}=build();assert.equal(p.Wx.length,3);assert.ok(p.Wx.every(r=>r.length===4));
  assert.equal(p.Wh.length,3);assert.ok(p.Wh.every(r=>r.length===3));assert.equal(p.b.length,3);
});
test('Case 6: seed reproduces embeddings, weights, states for both initializations',()=>{
  for(const init of ['zeros','random'])assert.deepEqual(build('same same',8,4,3,42,init),build('same same',8,4,3,42,init));
  assert.notDeepEqual(build().parameters,build('test',8,4,3,43).parameters);
});
test('Independent hand calculation validates recurrence and bias through padding',()=>{
  const p={Wx:[[1,2],[-1,0.5]],Wh:[[0.5,0],[0,-1]],b:[0.1,-0.2],h0:[0.2,-0.3]};
  const steps=forward(['a','<PAD>'],new Map([['a',[0.4,-0.2]],['<PAD>',[0,0]]]),p);
  assert.ok(Math.abs(steps[0].h[0]-Math.tanh(0.2))<1e-14);
  assert.ok(Math.abs(steps[0].h[1]-Math.tanh(-0.4))<1e-14);
  assert.ok(Math.abs(steps[1].h[0]-Math.tanh(0.5*Math.tanh(0.2)+0.1))<1e-14);
  assert.ok(Math.abs(steps[1].h[1]-Math.tanh(-Math.tanh(-0.4)-0.2))<1e-14);
});
test('Empty and whitespace-only inputs pad safely; exact length needs no handling',()=>{
  assert.deepEqual(preprocess(' \n\t ',1).tokens,['<PAD>']);assert.deepEqual(preprocess(' a\n b\t c ',3).tokens,['a','b','c']);
});
test('Initialization bounds, zero bias, and reproducibility for all modes',()=>{
  for (const mode of ['small','xavier','zeros']) {
    const p=initialize(4,3,42,'random',mode);
    assert.deepEqual(p,initialize(4,3,42,'random',mode));
    assert.deepEqual(p.b,[0,0,0]);
    assert.deepEqual(p.h0,initialize(4,3,42,'random','small').h0);
    const bx=mode==='xavier'?Math.sqrt(6/7):mode==='small'?0.25:0;
    const bh=mode==='xavier'?1:mode==='small'?0.5/Math.sqrt(3):0;
    assert.ok(p.Wx.flat().every(v=>Math.abs(v)<=bx));
    assert.ok(p.Wh.flat().every(v=>Math.abs(v)<=bh));
    if(mode!=='zeros') assert.notDeepEqual(p.Wx,initialize(4,3,43,'random',mode).Wx);
  }
  const small=initialize(4,3,42,'zeros','small'), xavier=initialize(4,3,42,'zeros','xavier');
  assert.ok(Math.abs(xavier.Wx[0][0]/small.Wx[0][0]-Math.sqrt(6/7)/0.25)<1e-12);
  assert.ok(Math.abs(xavier.Wh[0][0]/small.Wh[0][0]-Math.sqrt(6/6)/(0.5/Math.sqrt(3)))<1e-12);
});
test('Zero weights yield zero states even with random h0 and nonzero embeddings',()=>{
  const tokens=['cat','cat','<PAD>'], map=embeddings(tokens,4,42);
  const p=initialize(4,3,42,'random','zeros');
  assert.ok(p.h0.some(v=>v!==0));assert.ok(map.get('cat').some(v=>v!==0));
  assert.ok(forward(tokens,map,p).every(s=>s.h.every(v=>v===0)));
});
