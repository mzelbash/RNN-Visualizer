# RNN Visualizer

**Dr. Elbasheer**

An interactive educational application that shows how a vanilla recurrent neural network processes a sentence one token at a time. Students can explore embeddings, recurrent memory, shared weights, and optional predictions through actual numerical calculations.

All calculations run in the browser. No backend, model training, or build step is required.

## Try it live   

https://mzelbash.github.io/RNN-Visualizer/

## Run locally

From the project folder containing `rnn/`, run:

```sh
python -m http.server 8000
```

Open [RNN Visualizer locally](http://localhost:8000/rnn/).

Use an HTTP server instead of opening the HTML file directly. The application uses JavaScript modules. MathJax is loaded from a CDN for equation typesetting; readable HTML equations remain available without it.

## Setup

1. Enter a sentence. The default is `The cat sat on the mat`.
2. Set the maximum sequence length, from 1 to 12.
3. Choose embedding and hidden sizes, from 1 to 128.
4. Choose an initial hidden state: Zeros or Random.
5. Choose a weight initialization: Small Random, Xavier Uniform, or Zeros.
6. Set the seed to make the example reproducible.
7. Optionally enable an output layer and choose its number of classes.
8. Select **Build RNN** to apply the settings.

The activation is fixed to tanh. Bias vectors are initialized to zero. Rebuilding with the same settings and seed reproduces the same embeddings, weights, and hidden states.

## Teaching Flow

1. Enter a sentence.
2. Observe whitespace tokenization.
3. Observe padding or truncation.
4. Inspect simulated embeddings.
5. Observe the unrolled RNN.
6. Step through one timestep at a time.
7. Inspect the current input x_t, previous hidden state h_(t-1), and new hidden state h_t.
8. Open the numerical calculation.
9. Emphasize that the same RNN weights are reused across all timesteps.

## Navigate the visualization

- **Start:** No tokens have been processed in the displayed walkthrough and no results are revealed.
- **Next Step:** Process the next token and reveal its hidden state and calculation.
- **Previous Step:** Return to the previous timestep. From timestep 1, return to Start.
- **Run All:** Restart at Start and reveal one timestep every 1.5 seconds.
- **Pause:** Stop automatic playback.
- **Reset:** Return to Start without changing model parameters.
- **Click a token or cell:** Inspect that timestep directly. Its state includes the preceding recurrent calculations.
- **Hide controls / Hide details:** Collapse either side panel to give the canvas more space. Use the same buttons to restore it.
- **Highlight Shared Weights:** Highlight all cells together to emphasize parameter reuse.

Words and embeddings enter the cells from below. Hidden states pass horizontally between cells. The selected hidden-state vector appears beneath its timestep label. The **Inside this timestep** panel shows the numerical calculation below the diagram.

Neuron View shows individual values when both embedding and hidden sizes are at most 8. Larger sizes automatically use Vector View. Scroll horizontally to explore longer sequences and vertically within long vectors to inspect all values.

## Sequence handling and embeddings

Tokenization splits on whitespace and preserves case and punctuation. `The` and `the` are different tokens. Repeated occurrences of the same token share an embedding.

Short sequences receive `<PAD>` tokens. Long sequences are truncated after the selected maximum length. Empty input produces an all-PAD sequence. The reserved `<PAD>` token always has a zero embedding, including when entered directly.

Embeddings are simulated values for teaching, not pretrained or learned representations. PAD positions still undergo recurrent computation, so their hidden states can change. The application does not apply recurrent masking.

## RNN calculation

At each timestep:

```text
h_t = tanh(Wx x_t + Wh h_(t-1) + b)
```

The same Wx, Wh, and b are reused at every timestep. Column-vector dimensions are:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| x_t | E × 1 | Current token embedding |
| h_t | H × 1 | Hidden state |
| Wx | H × E | Input weights |
| Wh | H × H | Recurrent weights |
| b | H × 1 | Bias |

E is embedding size and H is hidden size. Calculations use full precision internally; displayed values are generally rounded to three decimals. Small differences when adding displayed numbers are due to rounding.

**Show Calculation** reveals the input contribution, recurrent contribution, bias, preactivation, and tanh result. Select a hidden unit to inspect every multiplication, or expand the shared parameter tables to inspect the complete matrices.

## Weight initialization

| Option | Wx range | Wh range |
| --- | --- | --- |
| Small Random | ±0.5 / sqrt(E) | ±0.5 / sqrt(H) |
| Xavier Uniform | ±sqrt(6 / (E + H)) | ±sqrt(6 / (2H)) |
| Zeros | All zeros | All zeros |

Random values are sampled uniformly within the specified bounds. Bias is zero for every option. Random initial hidden-state values lie between -0.25 and 0.25. Embeddings and initial hidden states use separate seeded streams from the weights.

With zero weights and zero bias, every computed hidden state is zero, even when the initial hidden state is random.

## Optional output layer

**Output Mode** offers:

- **None:** Show only recurrent processing.
- **Every timestep:** Produce a prediction above each real-token cell.
- **Final real token only:** Produce one prediction from the last retained non-PAD token.

Output size O can be from 2 to 8 classes and is independent of hidden size H. With three classes in every-timestep mode, illustrative labels are Noun, Verb, and Other. Other configurations use generic class labels.

**These are untrained demonstrations. The labels are not meaningful grammatical predictions.**

The output layer computes:

```text
z_out = Wy h_t + by
 y_t  = softmax(z_out)
```

Wy has shape O × H, while by, scores, and probabilities have shape O × 1. Output weights use a separate seeded stream and the selected initialization method. Output bias is zero. The same output weights are reused across timesteps.

Click an output to inspect its hidden vector, scores, probabilities, shared output weights, and per-class arithmetic. The largest probability selects the label; ties select the first class.

Hidden states carry memory forward. Predictions do not feed into the next cell. A prediction uses only the current token and preceding tokens, not future tokens. PAD outputs are ignored, and an all-PAD sequence has no predictions. Enabling outputs does not alter the recurrent states.

## Deploy to GitHub Pages

Keep this structure when uploading:

```text
README.md
rnn/
  index.html
  styles.css
  package.json
  js/
  tests/
```

1. Upload the complete `rnn/` folder and this README to your repository, preserving the folder structure.
2. Commit the changes to the branch you want to publish.
3. In the repository, open **Settings > Pages**.
4. Choose **Deploy from a branch**, select the published branch, and select **/ (root)**.
5. Save and wait for deployment to complete.
6. Open the published Pages address with `/rnn/` appended.

If Pages is already configured to publish that branch and folder, uploading the changes triggers an update. No npm installation or build command is needed for deployment. This README is the only documentation file required; there is no dependency on a README inside `rnn/`.

## Source structure

| File | Purpose |
| --- | --- |
| `rnn/index.html` | Layout and controls |
| `rnn/styles.css` | Theme and responsive layout |
| `rnn/js/sequence.js` | Tokenization, padding, and truncation |
| `rnn/js/random.js` | Deterministic random-number generation |
| `rnn/js/embeddings.js` | Simulated token embeddings |
| `rnn/js/math.js` | Weight initialization and RNN forward pass |
| `rnn/js/visualization.js` | SVG network rendering |
| `rnn/js/render.js` | Sequence, vector, and matrix rendering |
| `rnn/js/details.js` | Detailed hidden-unit calculations |
| `rnn/js/operation.js` | Main timestep calculation panel |
| `rnn/js/output.js` | Optional output layer and inspection |
| `rnn/js/app.js` | Setup, selection, playback, and panel controls |
| `rnn/tests/rnn.test.js` | Numerical validation tests |

## Run tests

Using a current Node.js LTS installation, run:

```sh
cd rnn
npm test
```

The 12 tests cover padding, truncation, repeated-token embeddings, large dimensions, matrix shapes, reproducibility, a hand-calculated recurrence, initialization bounds, zero weights, output selection, and stable softmax probabilities.

## Scope

This application is a teaching tool for a vanilla RNN forward pass with an optional output layer. It does not implement training, backpropagation, loss functions, optimizers, LSTM, GRU, attention, or transformers.

## License

See the repository's `LICENSE` file.
