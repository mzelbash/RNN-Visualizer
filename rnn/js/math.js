import { random } from './random.js';
export const matvec = (matrix, vector) => matrix.map(row => row.reduce((sum, value, j) => sum + value * vector[j], 0));
export function initialize(E, H, seed, initial, weights = 'small') {
  if (!['small', 'xavier', 'zeros'].includes(weights)) throw new Error('Unknown weight initialization');
  const rng = random(seed, 'parameters');
  const vector = (n, scale) => Array.from({ length: n }, () => (rng() * 2 - 1) * scale);
  // Xavier uniform uses sqrt(6 / (fan_in + fan_out)) for each matrix.
  const matrix = (columns, bound) => Array.from({ length: H }, () => weights === 'zeros' ? Array(columns).fill(0) : vector(columns, bound));
  const Wx = matrix(E, weights === 'xavier' ? Math.sqrt(6 / (E + H)) : 0.5 / Math.sqrt(E));
  const Wh = matrix(H, weights === 'xavier' ? Math.sqrt(6 / (H + H)) : 0.5 / Math.sqrt(H));
  const b = Array(H).fill(0);
  const initialRng = random(seed, 'initial-state');
  const h0 = Array.from({ length: H }, () => initial === 'random' ? initialRng() * 0.5 - 0.25 : 0);
  return { Wx, Wh, b, h0 };
}
// Column-vector convention: (H × E)(E × 1) + (H × H)(H × 1) + (H × 1).
export function forward(tokens, embeddingMap, parameters) {
  let previous = parameters.h0;
  return tokens.map((token, index) => {
    const x = embeddingMap.get(token);
    const inputTerm = matvec(parameters.Wx, x);
    const recurrentTerm = matvec(parameters.Wh, previous);
    const z = inputTerm.map((v, i) => v + recurrentTerm[i] + parameters.b[i]);
    const h = z.map(Math.tanh);
    const step = { token, t: index + 1, x, previous, inputTerm, recurrentTerm, z, h };
    previous = h;
    return step;
  });
}
