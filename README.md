# ICL Experiment Framework

An experiment framework for evaluating in-context learning (ICL) example
selection with local OpenAI-compatible model servers. Implemented selectors
include Random, BM25, KATE, CEIL, and GenICL.

## What is included

- Unified dataset adapters, prompt construction, metrics, and result logging.
- Qwen and Llama inference through a local vLLM OpenAI-compatible endpoint.
- Reproducible example selection for Random, BM25, and KATE.
- An adapted CEIL training pipeline using a frozen Qwen teacher.
- Unit tests for evaluation contracts, data filtering, and CEIL mathematics.
- Compact `summary.json` files from experiment runs.

Downloaded models, generated supervision caches, checkpoint weights, raw
per-sample outputs, and logs are intentionally excluded from Git.

## Environment

Create the environment from:

```bash
conda env create -f environment-vllm.yml
conda activate vllm
```

Model checkpoints are expected under `models/` by default. They can also be
changed in `scripts/evaluation/config.py`.

## Start a local model server

Example for Qwen2.5-7B-Instruct:

```bash
vllm serve models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --enable-prefix-caching
```

## Evaluation

Run a zero-shot evaluation:

```bash
python scripts/evaluate.py \
  --method zero_shot \
  --models qwen \
  --datasets sst5 banking77 cmsqa \
  --output-dir results/zero_shot_example
```

Run a BM25 evaluation:

```bash
python scripts/evaluate.py \
  --method bm25 \
  --models qwen \
  --datasets sst5 \
  --num-shots 4 \
  --output-dir results/bm25_example
```

See [`scripts/README.md`](scripts/README.md) for the unified evaluation CLI.

## CEIL training

The CEIL pipeline contains six stages:

1. Construct singleton candidates.
2. Score singleton candidates with a frozen Qwen teacher.
3. Train the EPR initializer.
4. Construct candidate demonstration sets.
5. Score complete demonstration sets with Qwen.
6. Train the conditional-DPP retriever and evaluate it at different shot counts.

Commands and checkpoint metadata are documented in
[`scripts/ceil_training/README.md`](scripts/ceil_training/README.md).

This implementation is a project-specific Qwen adaptation. Its data budgets,
teacher model, candidate settings, and dataset splits may differ from the
original paper configuration and should not be described as an exact numerical
reproduction of the paper.

The upstream implementation used for reference is:

- Repository: <https://github.com/HKUNLP/icl-ceil>
- Commit: `e23c539810f4275e1c65f7a3c91b3594aa31c575`
- Paper: *Compositional Exemplars for In-context Learning* (ICML 2023)

## Tests

```bash
python -m unittest discover -s tests -v
```

## Data and artifacts

Hugging Face datasets are downloaded by the dataset adapters. Banking77 files
are expected under `data/banking77/`. Generated data and model weights are not
tracked in this repository. Compact experiment summaries and checkpoint
metadata are retained to document configurations and reported metrics.

## License

No project-level license has been selected yet. Third-party datasets, models,
and upstream implementations remain subject to their own licenses.
