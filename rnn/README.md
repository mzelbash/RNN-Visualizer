# RNN Visualizer — Version 1

A static teaching application for the vanilla RNN forward pass. Open `rnn/` on your static host. The existing FCNN application remains at the repository root.

## Run locally

From the repository root:

```sh
python -m http.server 8000
```

Visit http://localhost:8000/rnn/. ES modules require an HTTP server; do not open the HTML using `file://`. No npm installation or build step is required. MathJax 3.2.2 is loaded from jsDelivr for equation typesetting; readable HTML equations remain available without it. All model computations run locally in the browser.

## Deploy to GitHub Pages

1. Commit and push the `rnn/` directory and the updated root `index.html` and `README.md` to your repository.
2. In GitHub Settings → Pages, choose “Deploy from a branch,” your published branch (typically `main`), and `/ (root)`.
3. The FCNN app remains at `https://mzelbash.github.io/NN-Visualizer/`; RNN Visualizer is served at `https://mzelbash.github.io/NN-Visualizer/rnn/` once deployment completes.

This implementation does not publish or push automatically.

## Teaching Flow

The title and Dr. Elbasheer attribution live in the controls panel. Use **Hide controls** and **Hide details** above the canvas to independently collapse either sidebar; the same buttons restore them without resetting the model. Token embeddings enter each cell from below; recurrent states travel horizontally. Timestep labels sit beneath the words, with a hidden-state readout beneath the selected timestep. This readout is not an output layer; the space above cells is left clear for future optional outputs. The **Inside this timestep** panel stays beneath the network, showing color-coded input and recurrent contributions, bias, preactivation, and the new hidden state on every selection or step.

1. Enter sentence.
2. Observe tokenization.
3. Observe padding or truncation.
4. Inspect simulated embeddings.
5. Observe the unrolled RNN.
6. Step through one timestep at a time.
7. Inspect x_t, h_(t−1), and h_t.
8. Open the numerical calculation.
9. Emphasize that the same RNN weights are reused across all timesteps.

## Model conventions

Whitespace splitting preserves case and punctuation; `The` and `the` are distinct tokens. Empty input produces an all-PAD sequence. `<PAD>` is reserved and always has a zero embedding, including when typed explicitly. Maximum length accepts integers 1–12; embedding and hidden sizes accept integers 1–128. Seeds are unsigned 32-bit integers.

Column vectors are used throughout: x has shape E × 1; h and b have shape H × 1; Wx has shape H × E; Wh has shape H × H. Each timestep computes `h = tanh(Wx x + Wh h_previous + b)` with JavaScript double precision. Only presentation is rounded to three decimals. Long vectors scroll vertically and complete weight matrices are available on demand. The unit selector shows every multiplicative term for any hidden unit.

Embeddings are simulated uniform values in [−1, 1). Named seeded PRNG streams keep embeddings independent of token order and parameters independent of sentence content. Weight Initialization offers Small Random (default), Xavier Uniform, and Zeros. Small Random uses Wx values uniform in ±0.5/√E and Wh in ±0.5/√H. Xavier Uniform uses bounds ±√(6/(E+H)) for Wx and ±√(6/(2H)) for Wh. Zeros sets both matrices to zero. Bias is zero in every mode; optional random initial states remain in ±0.25. Embeddings and initial states are independent of the weight initialization choice. Zero weights produce zero hidden states even with a random initial state. These are small illustrative values, not trained parameters. Rebuilding the same settings reproduces the same model.

PAD timesteps still apply recurrent weights and bias; no masking is performed. Future state values remain unrevealed until selected. Clicking a later cell inspects its correctly computed state, including preceding recurrent calculations. Build and Reset return to Start, with no token selected or result revealed. Next Step processes the first token; Previous Step from timestep 1 returns to Start. Run All begins at Start and processes the first token after 1.5 seconds, then advances every 1.5 seconds; Pause or manual selection stops playback.

Neuron View displays one circle per input/hidden value when both dimensions are at most 8. Larger dimensions automatically select Vector View. The horizontally scrollable SVG preserves label readability for up to 12 steps; controls and details stack on small screens. Cells support keyboard Enter/Space, and sequence buttons provide a second navigation route.

The base visualizer excludes training, losses, optimizers, backpropagation, masking, LSTM, GRU, attention, and transformers. An optional untrained output-layer extension is now included below.

## File structure

```text
rnn/
  index.html             Semantic layout and controls
  styles.css             Responsive classroom-friendly styling
  js/
    sequence.js          Whitespace tokenization, padding, truncation
    random.js            Seeded named random streams
    embeddings.js        Shared per-token simulated embeddings
    math.js              Initialization and actual forward pass
    visualization.js     SVG unrolled network, automatic view selection
    render.js            Sequence, embedding, vector and matrix rendering
    details.js           Calculation inspection and hidden-unit arithmetic
    operation.js         Prominent color-coded classroom calculation
    app.js               Build, selection, reset and playback interaction
  tests/rnn.test.js       Six required cases and independent arithmetic checks
```

## Validation

Requires Node.js 18 or later:

```sh
cd rnn
npm test
```

Tests cover all six specified cases, a hand-calculated two-unit recurrence, PAD recurrence, random initialization reproducibility, empty inputs and whitespace handling. For UI verification: build defaults, inspect embeddings, click cells, step backward/forward, run and pause, reset, highlight shared weights, expand calculations and parameters, choose hidden size 128, and resize to mobile width.

Uses the repository's MIT license.

## Optional output-layer extension

Output Mode defaults to None, preserving the original forward-pass view. Every timestep adds a shared linear classification head above each real token; Final real token only uses the last non-PAD token retained after truncation. PAD outputs are ignored, and empty input has no output. Output size is independent of hidden size (2–8 classes). Three-class per-token examples use Noun / Verb / Other labels; these are untrained demonstrations, not meaningful grammatical predictions. Other configurations use generic class labels.

The output calculation is `z_out = Wy h + by`, then stable softmax. Wy has shape O × H and by has shape O × 1. A separate seeded random stream initializes Wy using the selected Small Random, Xavier Uniform, or Zeros method; by is zero. Toggling output modes does not change embeddings or recurrent states. Outputs never feed back into recurrence. Click an output to inspect its full hidden vector, scores, probabilities, output matrices, and per-class arithmetic below the existing timestep calculation. Outputs are unrevealed in Start; stepping reveals them sequentially. The optional head adds no training or loss calculation.

