# Korean Math TDCS + Recirculation

Reproducible research code for Korean mathematical-reasoning post-training with:

1. untouched `LGAI-EXAONE/EXAONE-4.0-1.2B` evaluation;
2. random-sampling LoRA SFT;
3. Transfer-aware Dynamic Curriculum Sampling (TDCS) LoRA SFT; and
4. fixed, training-free Recirculation applied to the best post-trained checkpoint.

The package is the source of truth. The notebooks are intentionally thin launchers.

## Installation

Python 3.10+ and `transformers>=4.54` are required. On a fresh Colab/runtime:

```bash
git clone <REPO_URL> korean-math-tdcs
cd korean-math-tdcs
pip install -e .
```

The primary hardware target is one NVIDIA RTX PRO 6000 Blackwell (96 GB). If batch 32 does not
fit on another GPU, reduce `training.micro_batch_size`; the code accumulates micro-batches inside
each fixed effective optimizer batch of 32.

## Reproduction

Audit the dataset and exact EXAONE serialization first:

```bash
python scripts/analyze_difficulty.py --config configs/tdcs.yaml
```

This writes `results/dataset_audit/dmath_tdcs_audit.json`, checks every final serialized sequence,
and prints three completed assistant examples. Gold reasoning must appear inside the native
`<think>...</think>` block.

Run the four experiment stages:

```bash
# 1. Base evaluation
python scripts/evaluate.py --config configs/baseline.yaml

# 2. Random LoRA SFT, then evaluate its adapter
python scripts/train_sft.py --config configs/sft.yaml
python scripts/evaluate.py --config configs/baseline.yaml \
  --set model.adapter=results/sft/run_001/adapter \
  --set output.results_path=results/sft/evaluation.json

# 3. TDCS LoRA SFT, then evaluate its adapter
python scripts/train_tdcs.py --config configs/tdcs.yaml
python scripts/evaluate.py --config configs/baseline.yaml \
  --set model.adapter=results/tdcs/run_001/adapter \
  --set output.results_path=results/tdcs/evaluation.json

# 4. Put the winning adapter in recirculation.yaml, tune only on validation,
# copy the selected source/destination/alpha into the same config, then evaluate.
python scripts/sweep_recirculation.py --config configs/recirculation.yaml
python scripts/evaluate_recirculation.py --config configs/recirculation.yaml
```

Any YAML value can be overridden with repeatable `--set dotted.key=value` arguments. Use
`evaluation.<benchmark>.max_samples` and a smaller training configuration for smoke tests; do not
present such runs as the primary experiment.

Ordinary evaluation uses batched generation (`evaluation.batch_size: 64`) to saturate large GPUs.
If a smaller GPU runs out of memory, reduce it without changing generation semantics:

```bash
python scripts/evaluate.py --config configs/baseline.yaml --set evaluation.batch_size=16
```

Fixed Recirculation remains batch size 1 because its state is propagated sequentially between tokens.
Evaluation predictions are appended continuously and resumed by UID after an interruption. Set
`evaluation.resume=false` (or delete the corresponding `*_predictions.jsonl`) when intentionally
changing a model, prompt, parser, or generation configuration while reusing the same results path.

## Fixed experiment definition

- Training data: `keunhyeung/dmath-ko-reasoning-dpo`, `reasoning-sft`.
- Difficulty: `operator_count = len(gold_trace)` and D1/D2/D3/D4/D5 = `1/2/3/4/5+`.
- Budget: 2,708 rows × 4 epochs = 10,832 ordinary example draws and 339 optimizer steps at
  effective batch 32. Probe batches are tracked separately.
- SFT target: EXAONE native reasoning serialization with `skip_think=False`; user/prompt tokens are
  masked from loss.
- Transfer matrix: `Re(i,j) = (g_i^T g_j)/(g_i^T g_i)`, where rows are affected/evaluated
  difficulties and columns are training/source difficulties. Only trainable LoRA gradients are used.
- Transfer probing: 8 examples from each difficulty at initialization and every 25 optimizer steps.
- Curriculum: one monotonic D1→D5 pass using equal partitions of the resolved optimizer-step budget.
- Evaluation: the same reasoning prompt and benchmark-specific answer parser for every method.
- Recirculation: fixed only, with serialized token-by-token prefill and previous-token deep-state
  feedback. It must be selected on validation, never on final test data.

The TDCS paper defines the sigmoid and exponential forms but does not report numerical values for
sigmoid slope/midpoint or replay decay. It also says that “a portion” of the current-level mass is
reassigned to qualifying harder levels without specifying the portion. The primary adaptation makes
these implementation choices explicit in `configs/tdcs.yaml`:

```yaml
sigmoid_alpha: 10.0
sigmoid_beta: 0.5
replay_lambda: 1.0
harder_fraction: 0.10
```

The paper's Equation 10 calls the exponential coefficient `lambda`, while Algorithm 1 labels it
`beta`; this repository uses `replay_lambda`. These values are configurable and should be reported.
The primary learning rate is `1e-5` from the paper's main experimental description; its appendix
table instead reports `5e-5`.

## Results and logs

Training run directories contain resolved `config.yaml`, `adapter/`, `metrics.json`, and
`training_log.jsonl`. TDCS additionally records `transfer_history.jsonl` with matrices, sampling
probabilities, current stages, and the next batch's realized difficulty counts. Metrics separate:

- ordinary example draws and assistant target tokens;
- optimizer steps;
- probe examples, tokens, and backward passes;
- wall-clock time and peak GPU memory; and
- evaluation latency and generated tokens/second.

Prediction files contain the raw output, parsed answer, gold answer, and per-example latency so
parser behavior remains auditable.

## Benchmarks

- [HRM8K](https://huggingface.co/datasets/HAERAE-HUB/HRM8K), all five released subsets;
- [SNU Ko-GSM8K](https://huggingface.co/datasets/thunder-research-group/SNU_Ko-GSM8K); and
- [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU), explicit Math/STEM subject configs.

The relevant methods are [TDCS (arXiv:2608.17268)](https://arxiv.org/abs/2608.17268) and
[Recirculation (arXiv:2608.17981)](https://arxiv.org/abs/2608.17981). Recirculation layer numbers
are never copied from Gemma: `sweep_recirculation.py` first inspects the loaded EXAONE decoder depth
and constructs valid source/destination candidates.

## Verification

Fast, model-free tests cover difficulty mapping, exact budgets and stage boundaries, Relative
Transfer orientation, probability validity/direction, answer parsing, and Recirculation's disabled
path:

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

Full network/GPU smoke checks are intentionally separate because they download model and dataset
artifacts. Start those with `analyze_difficulty.py`, then a tiny overfit run via command-line
overrides before committing the full GPU budget.
