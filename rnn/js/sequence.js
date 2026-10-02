export function preprocess(sentence, maximum) {
  const original = sentence.trim() ? sentence.trim().split(/\s+/u) : [];
  const tokens = original.slice(0, maximum);
  const padCount = Math.max(0, maximum - tokens.length);
  return { original, tokens: [...tokens, ...Array(padCount).fill('<PAD>')], padCount,
    truncated: original.slice(maximum), maximum };
}
