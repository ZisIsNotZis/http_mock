# http_mock — learning to imitate toy protocol servers with a GRU seq2seq

**What:** Five Jupyter notebooks that (a) generate synthetic request/response
traces from two hand-written toy "mocks" — a REST-ish HTTP mock over relational
objects (`http_mock.ipynb`) and a line-oriented key-value store with arithmetic
commands (`http_mock_2/3/4.ipynb`) — and (b) train a small PyTorch GRU
encoder–decoder seq2seq to map a request string to the response string the mock
would have produced. In other words, a char-level imitation/simulation
experiment: can a sequence model learn to *be* the server?

**Why it exists:** Notes-to-self while exploring learned program/state-machine
simulation on synthetic traces. The interesting part is the protocol design and
the extra loss terms, not any result: `http_mock_3.ipynb` adds a triple loss
(target translation + two cycle/denoising reconstructions) and a cross-grid
`Seq2Rnn2Seq` that runs an RNN over a 2-D request/response token grid. Nothing
here was shipped, benchmarked, or used by anything; there are no saved outputs
and no recorded metrics.

**How to run:** there is no entrypoint, runner script, or dependency manifest.
The notebooks are standalone; the intended flow is to open one and "Run All":
```
jupyter lab        # then open http_mock_3.ipynb and Run All
```
This requires a CUDA GPU — `device='cuda'` is hardcoded in the model/`Lang`
constructors (`http_mock.ipynb`, `http_mock_2/3/4.ipynb`) and the training loops
call `.cuda()`. Not run here during migration (read-only review).

**Status:** **deprecated (~early-2020s GRU-seq2seq era) — kept as an archive.**
The PyTorch calls themselves are not deprecated (`nn.GRU`, `nn.Embedding`,
`OneCycleLR`, `functional.cross_entropy` are all supported), but the *approach* is
decades-of-iterations stale by 2026: char-level GRU encoder–decoder with no
attention, no tokenizer, no transformer, trained from scratch on toy synthetic
data, with no eval split, no metrics, no checkpoints. `http_mock_4.ipynb` is an
abandoned work-in-progress whose first cell is truncated mid-edit (an
unterminated string literal), and `box.ipynb` is a scratchpad. Treat the whole
directory as an honest archive entry, not a project to maintain.

**License:** no LICENSE file present. As the owner's own work, add an explicit
license (e.g. MIT) before publishing anything; until then, all rights reserved.
