// Independent named streams keep token embeddings stable across sentence order.
export function random(seed, name = '') {
  let state = seed >>> 0;
  for (const character of name) state = Math.imul(state ^ character.codePointAt(0), 16777619) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
