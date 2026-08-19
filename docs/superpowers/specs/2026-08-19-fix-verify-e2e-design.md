# Design: Fix & Verify BERT Ablation Factory (Runs End-to-End)

**Date:** 2026-08-19
**Status:** Approved design
**Repo:** `bert-ablation-factory`

## Goal

Make the `bert-ablation-factory` framework run reliably end-to-end. Fix broken
imports and config references, add a lightweight offline smoke-test suite, and
prove every CLI entry point runs. Verification is done with tiny synthetic data
and tiny local checkpoints (no network, no heavy downloads, minutes not hours).

The goal explicitly is **not** a full open-source polish pass (packaging exports,
LICENSE, GitHub Actions CI, whole-dataset integration tests are out of scope).

## Confirmed defects (source of motivation)

1. **Missing module.** `cli/pretrain.py`, `cli/finetune_classification.py`, and
   `cli/finetune_qa.py` all do `from ..data.tokenization import build_tokenizer`,
   but no `bert_ablation_factory/data/tokenization.py` exists. All three CLIs
   fail on import.
2. **Broken README command.** References `configs/pretrain/mlm_only_base.yaml`;
   the real file is `mlm_no_nsp_base.yaml`.
3. **README claims `tests/` + "Unit tests"** exist; there is no `tests/` dir.
4. Model construction in every CLI is hardwired to
   `from_pretrained(<real HF model>)`, which blocks offline/tiny testing; the
   GLUE/SQuAD data path is hardwired to `load_dataset` (network).

## Approach (chosen: A)

Surgical fix + offline smoke tests. Add the missing tokenization module, build a
`tests/` suite driven by tiny local checkpoints and tiny configs, add a
user-facing `DATA.source` dataset option, and clean up the docs. No invasive
refactor of model construction.

## Design

### 1. `data/tokenization.py` (critical missing module)

Create `bert_ablation_factory/data/tokenization.py` exposing
`build_tokenizer(cfg) -> transformers.PreTrainedTokenizer`:

- Reads `cfg["MODEL"]["name"]`.
- Returns `BertTokenizer.from_pretrained(name)`.
- Raises `ValueError`/`FileNotFoundError` with a clear message if the name is
  missing or load fails.
- No new dependency (tokenizers ships with `transformers`).

All three CLIs already call `build_tokenizer(cfg)`, so this unblocks every entry
point unchanged.

### 2. Offline smoke-test infrastructure (`tests/`)

- **`tests/conftest.py`** provides fixtures that need no network:
  - `tiny_bert_dir(tmp_path)`: build a tiny `BertConfig` (e.g. 2 hidden layers,
    `hidden_size=32`, `vocab_size=30522`, small `max_position_embeddings`),
    instantiate small random-weight per-head checkpoints
    (`BertForPreTraining`, `BertForMaskedLM`, `BertLMHeadModel`,
    `BertForSequenceClassification`, raw `BertModel`), and
    `save_pretrained()` each to a temp dir. `from_pretrained(<tmpdir>)` then
    loads them offline, exercising the real CLI code paths.
  - `tiny_pretrain_cfg` / `tiny_glue_cfg` / `tiny_squad_cfg`: tiny YAML configs
    that set `MODEL.name` to the tiny dir and override `TRAIN` to a few steps /
    one epoch with a temp `OUTPUT_DIR` and `DATA.source: synthetic`.
- **Component unit tests** (pure; no weights):
  - `data/collators.py`: MLM 80/10/10 vs 100% masking, LTR right-shift labels,
    label `-100` construction.
  - `modeling/heads.py`: `ClassificationHead` / `SpanHead` forward shapes, with
    and without BiLSTM.
  - `modeling/objectives.py`: `LossCombine` for MLM/NSP/LTR.
  - `trainer/eval.py`: `accuracy` and `squad_em_f1`.
  - `registry.py`: register / get / duplicate-key error / keys.
  - `utils/io.py`: `load_yaml` (valid, missing file, bad YAML), `merge_dict`.
  - `data/tokenization.py`: `build_tokenizer` happy path + error paths.

### 3. `DATA.source` dataset injection (chosen: option b)

Add a `source` key in `DATA` to `data/glue.py` and `data/squad.py`:

- **`data/glue.py` — `build_glue_task`:**
  - `source: "hf"` (default) → current `load_dataset("glue", task)`.
  - `source: "json"` → load from `DATA.train_path` / `DATA.dev_path` (any
    HF-loadable local files with the task's expected columns).
  - `source: "synthetic"` → generate small in-memory examples for the active
    task (used by smoke tests; no files, no network).
- **`data/squad.py` — `load_squad_v1`:** same three modes.
  - `json` → local SQuAD-style data (`question`, `context`, `answers`).
  - `synthetic` → tiny generated QA set.
- Validate that files exist for `json`; raise clear errors otherwise.
- CLIs are unchanged (they already route through these functions); only config
  keys are added. Default (`hf`) behavior is identical, so existing configs are
  backward-compatible.

### 4. CLI end-to-end smoke tests (the "does it run" proof)

Each runs the real `main()` entry with a tiny config and tiny step counts:

- `test_pretrain_cli` for each ablation (`mlm_nsp`, `mlm_only`, `ltr`) — asserts
  it trains, logs, and writes artifacts under `OUTPUT_DIR`.
- `test_finetune_classification_cli` — tiny synthetic SST-2 dataset, few epochs,
  asserts run dir / `best.pt` written.
- `test_finetune_qa_cli` — tiny synthetic SQuAD-style data, asserts weights
  saved.
- `test_checkpoint_roundtrip` — save then load restores model/optimizer/epoch/
  step.

Concretely, these tests import and invoke each CLI's `main` (or its internal
entry) with the tiny config so the full real pipeline (tokenizer → data → 
model build → train loop → checkpoint) is exercised.

### 5. Docs cleanup (minimal, in-scope)

- Fix broken README refs: `mlm_only_base.yaml` → `mlm_no_nsp_base.yaml`.
- Add a "Running the tests" section (`pytest`, and how to run the tiny
  entrypoints).
- Correct the Project Structure tree: add `data/tokenization.py` and `tests/`.
- Fix title casing ("BERT Ablation Factory").
- Document the new `DATA.source` option (hf / json / synthetic).

### 6. Test configuration

- Add `pytest.ini` (or `[tool.pytest.ini_options]` in a minimal
  `pyproject.toml`) so `pytest` collects `tests/` and the package imports
  cleanly. No packaging exports, LICENSE, or CI beyond this.

### 7. Verification (evidence before claims)

- Run `pytest` — all tests green.
- Run each CLI on the tiny config to confirm the real entrypoints execute
  (pretrain ×3 ablations, finetune classification, finetune QA).
- Report the exact commands and their output as evidence.

## Out of scope

- Packaging/`pip install .` distributions, publishing.
- LICENSE / CONTRIBUTING / GitHub Actions CI.
- Multi-GPU, distributed, or long-duration training.
- Real-dataset integration training (network downloads).
- Non-BERT tasks, or new model architectures.

## Success criteria

- `pytest` passes (no network access required).
- Each of the three CLIs' real entry path runs to completion on a tiny config.
- README commands/structure are correct and match the repo.
- No existing default behavior changed (HF + real-model flow still works).