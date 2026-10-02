import { random } from './random.js';
export function embeddings(tokens, size, seed) {
  return new Map([...new Set(tokens)].map(token => {
    const rng = random(seed, `embedding:${token}`);
    return [token, Array.from({ length: size }, () => token === '<PAD>' ? 0 : rng() * 2 - 1)];
  }));
}
