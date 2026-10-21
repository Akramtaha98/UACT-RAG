# UACT-RAG

Research code for **UACT-RAG: Cost-Aware Evidence Uncertainty Estimation and Adaptive Retrieval for Trustworthy Engineering Decision Support**.

A controller decides, at each step, whether to **ANSWER**, **RETRIEVE** more evidence, or **ABSTAIN**. It uses a calibrated estimate of the *evidence state* (sufficient, partial, irrelevant, missing, contradictory) and a cost-sensitive policy trained on measured action utilities. A claim-level verifier checks each answer against its cited passages.

> **Status: research code, no results yet.** Nothing here has been run on a GPU. The CPU smoke test checks wiring only. Every number in the paper must come from your own logged runs on the frozen test split.

## What is in the repo

| Path | Purpose |
| --- | --- |
| `uact/retrieval.py` | BM25 (`bm25s`) + BGE-M3/FAISS + reciprocal rank fusion + cross-encoder reranker |
| `uact/features.py` | Evidence features z = [relevance, coverage, contradiction, answerability] |
| `uact/estimator.py` | Five-state MLP estimator + temperature scaling (three-state variant for ablation A7) |
| `uact/controller.py` | Per-action utility regressors, abstention margin, threshold / fixed-budget controllers (A5, A2) |
| `uact/verifier.py` | Claim-level NLI verifier |
| `uact/systems.py` | UACT-RAG and baselines B0-B9, scoring |
| `uact/metrics.py`, `uact/report.py` | ECE, Brier, NLL, selective risk, AURC / E-AURC, bootstrap CIs, McNemar, Holm |
| `scripts/` | Data prep, indexing, state-data creation, training, evaluation, analysis |
| `run_all.sh` | The full pipeline in order |
| `tests/smoke_test.py` | CPU wiring test with stub models |

## Pull and set up on the H200

```bash
git clone https://github.com/Akramtaha98/UACT-RAG.git
cd UACT-RAG
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # installs torch, vllm, sentence-transformers, bm25s, faiss-cpu, ...
python tests/smoke_test.py               # should print SMOKE TEST PASSED (CPU only, no downloads)
huggingface-cli login                    # needed for gated models (e.g. Llama); Qwen3-8B needs no login
```

Notes:

- `vllm` pins its own `torch` build. If pip complains, install `vllm` first, then the rest.
- On an H200 (141 GB) one process holds the generator (`vllm_gpu_mem_util: 0.45`, about 63 GB) plus the embedder, reranker and NLI model with room to spare. Lower or raise this in `configs/default.yaml`.
- Replace `faiss-cpu` with `faiss-gpu` only if it matches your CUDA build.
- Use `tmux` or `nohup` for long runs: `nohup bash run_all.sh data/techqa > run.log 2>&1 &`.

Pull later updates with `git pull`.

## Data

Datasets are **not** included and must be obtained under their own licences.

```bash
# 1. RAGBench (CC-BY-4.0 on its Hugging Face card; verify the dataset id and field names)
python scripts/prepare_data.py ragbench --subsets emanual techqa

# 2. IBM TechQA (registration required; no open licence was found; not redistributed here)
python scripts/prepare_data.py techqa --root /path/to/TechQA --technotes /path/to/TechQA/training_dev_technotes.json --out data/techqa
#    File names and JSON fields differ between releases: edit the lines marked [ADJUST] in scripts/prepare_data.py.
#    If no held-out eval file exists, dev is split 50/50 into val/test and a warning is printed. Report that in the paper.

# 3. Your engineering-manual collection (schema in the paper, Section 5.2)
python scripts/prepare_data.py engineering --questions my_questions.jsonl --corpus my_passages.jsonl --out data/engineering
#    Splits are by equipment family, so revisions of one family never straddle train / val / cal / test.
```

Build the retrieval indexes:

```bash
python scripts/build_index.py --data data/techqa
python scripts/build_index.py --data data/engineering
```

Create evidence-state training data. **Contradiction examples need human review:**

```bash
python scripts/make_state_data.py candidates --subsets emanual techqa --n 600
#  -> data/contradiction_candidates.jsonl : a qualified reviewer sets "reviewed": true/false on every line
python scripts/make_state_data.py build --subsets emanual techqa
python scripts/make_state_data.py techqa_retrieved --data data/techqa      # real retrieval-outcome labels
```

`--allow_unreviewed` exists for development only. Do not report results from a model trained on unreviewed contradictions.

## Run everything

```bash
bash run_all.sh data/techqa
```

This runs, in order: estimator training (full, A1, A7) -> action rollouts on train and val -> controller training and tuning on val -> baseline threshold tuning on val -> **frozen** test evaluation of all systems -> forced-answer runs for risk-coverage curves -> batch-size-1 latency runs -> budget sweep -> tables and paired statistics.

Outputs land in `runs/main/` (see `paths.work_dir`):

- `full/estimator_report.json`: macro-F1, NLL, Brier, ECE, AUROC / AUPRC for insufficient-evidence detection, confusion matrix, and the three baseline detectors (RQ2)
- `controller_grid.json`: every (lambda, mu, delta) grid point on validation
- `eval/<dataset>_test/results_table.md` and `.csv`, `risk_coverage.png`, `paired_stats.csv`, and per-question records (`*.jsonl`) for case studies

Run steps individually by reading `run_all.sh`; each script has a docstring with its arguments. For quick iteration, use `--split val --max_q 100`.

## System ids

| Id | System | Note |
| --- | --- | --- |
| B0 | No retrieval | |
| B1 | Standard RAG (dense top-k) | |
| B2 | Hybrid RAG (BM25 + dense + RRF + reranker) | |
| B3 | Adaptive-RAG-**style** | **Approximation**: zero-shot LLM complexity router, not the trained classifier of Jeong et al. |
| B4 | CRAG-**style** | **Approximation**: reranker-score evaluator, corpus re-retrieval instead of web search. Not the original CRAG. |
| B5 | Self-RAG | Not included. Run the official repo (MIT licence) with its specialised model on the same queries and corpus and drop the outputs into the same record format. |
| B6 | Fixed RAG + calibrated-confidence abstention | threshold tuned on val |
| B7 | Budget-matched multi-retrieval | no decision, always retrieves B times |
| B8 | Calibrated-confidence retrieve / answer / abstain controller | simple strong baseline in the spirit of Know Before You Fetch; a reimplementation, not the authors' code |
| B9 | LLM sufficiency-autorater abstention | in the spirit of Joren et al. (2025) |
| B10 | BeliefRAG-style controller | Not included: no public code was found. Reimplement from the paper if you want this head-to-head. |
| UACT | Full method | |
| A1-A7 | Ablations | A1 relevance-only features, A2 fixed budget, A3 no verifier, A4 no abstention (offline only), A5 fixed thresholds, A6 no cost penalty, A7 three-state estimator |

Baselines B0-B2, B3, B4, B7 and A4 receive the generator prompt that allows `INSUFFICIENT_EVIDENCE`, so they can abstain by prompting. This is deliberate (it matches the strongest prompt-only practice) and should be stated in the paper.

## Design choices you should know about

- **Splits**: train fits models and builds controller targets; val does early stopping and all tuning; cal fits temperature scaling; test is touched once, by the final evaluation. The pipeline never reads test outcomes before that.
- **Controller targets**: ANSWER and ABSTAIN utilities are measured. RETRIEVE utilities use a *reference continuation* (answer if P(sufficient) >= `tau_ref`, otherwise retrieve if budget remains, otherwise abstain). This is an approximation and a stated limitation.
- **Selection rule** for lambda, mu and the abstention margin: lowest mean cost subject to validation selective risk <= `--risk_target` (default 0.10) and coverage >= `--min_coverage` (default 0.50). Pre-register these values.
- **Risk-coverage curves and AURC** come from `--forced` runs (every question answered, ranked by the system's confidence). For UACT the confidence is the margin V(answer) - V(abstain).
- **Latency** is exact only with `--batch_size 1`; batched runs report time divided by batch size.
- **GPU memory**: `nvidia_smi_used_gb` is the whole device; `torch_peak_gb_this_process` does not include vLLM's own allocation.
- **Budget sweep** reuses the controller trained at the default `max_rounds`; retrain per budget for a strict comparison.
- **Correctness**: an LLM judge (same generator) or token-F1 (`eval.use_llm_judge`). Neither is ground truth: validate on the 150-250 example human audit.
- **Verifier**: DeBERTa-MNLI entailment of each claim against cited (or all) passages. It is an automated proxy; known to fail when model memory agrees with a bad passage.

## Things to verify before trusting results

1. RAGBench dataset id, split names and field names (`prepare_data.py ragbench`). The label-to-state rules in `make_state_data.py` are transparent but heuristic.
2. TechQA file names and fields (`prepare_data.py techqa`, lines marked `[ADJUST]`).
3. `bm25s` and `sentence-transformers` API versions (`uact/retrieval.py`, `uact/models.py`). The reranker output is sigmoid-normalised defensively.
4. The vLLM chat template for your generator (`enable_thinking=False` is passed for Qwen3).
5. Model licences (Llama 3.1 Community License has attribution and naming rules) and the licences of any manuals you include.

## Licence

Code: choose a licence before publishing (MIT or Apache-2.0 are common). Datasets and models keep their own licences.
