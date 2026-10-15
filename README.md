# RoleCue v0.1.0rc4 release candidate

RoleCue-Core contains an exactly specified symbolic role-cue generator, the prior-inventory exclusion list needed to rebuild its formal main experiment, frozen numerical snapshots, a small held-out example, and a separately downloadable compact evidence archive for all 135 main runs. RoleCue-Binding contains the frozen three-condition task data, full-information and partial-information baselines, same-suffix counterexamples, 45 per-seed formal evaluations/predictions, and a separate structural holdout. The resource assesses role inference under known cue observation laws; it does not model natural-language frequency distributions.

## Install and inspect

From this directory, use Python 3.10 or later:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[plots]'
rolecue verify --root binding/data
rolecue verify-release --root .
```

The Binding `manifest.json` covers the frozen data files. `release_manifest.json` covers all files inside this compact candidate. The separate Core evidence ZIP is described by `core/evidence_manifest.json` and its SHA-256 in `core/evidence_verification.json`. Synthetic data and derived prediction evidence are licensed under CC BY 4.0 (`LICENSE-DATA.txt`); original software is MIT (`LICENSE`). Both authors and their affiliations are recorded in `CITATION.cff`. No DOI has been assigned in this candidate.

## Reproduce a paper figure

```sh
rolecue figure --snapshots core/snapshots --output reproduced
```

This regenerates the Core main reliability-grid heatmap as vector PDF and PNG, plus `reliability_grid_values.json`. It uses frozen aggregate numbers, so no external training drive or neural checkpoint is required. Check the source hash and nine cells per architecture in the JSON. The original frozen aggregate is copied into `core/snapshots/main_aggregate.json` with personal paths removed from related provenance.

Recreate the Binding seed-level diagnostic and recompute its full analysis from released predictions:

```sh
rolecue binding-figure --analysis binding/analysis/analysis.json --output reproduced
python -m scripts.analyze_binding_formal --runs binding/results --data binding/data --output recomputed_binding
```

The recomputed `analysis.json` should match the supplied numeric values to floating-point precision (verified maximum absolute difference below `4e-18` across Python 3.10/3.11). The analysis separates five-seed intervals from event-group bootstrap intervals; its 1,000 resamples use a fixed seed.

## Rebuild and rescore the full Core main experiment

Download the separate release asset `rolecue_core_main_evidence_v0.1.0rc4.zip` beside this directory. It contains compact, per-example scoring fields for every held-out IID and four-way counterfactual prediction from all 135 Core main runs: 2.7 million IID rows and 10.8 million counterfactual rows. The archive is 179,948,970 bytes, SHA-256 `2e6a4ade3ae197d959110a5c10e71799bbaea9acc59bde8268aeda4ff38a548b`. Its manifest records the SHA-256 of each source JSONL file and each packed CSV member. The compact records omit redundant tokens and metadata, while preserving every field needed for the manuscript's main probability, conflict and entity-aligned intervention endpoints.

From this candidate directory, independently rescore every run and compare with the frozen run summary and aggregate table:

```sh
python -m scripts.core_evidence verify --archive ../rolecue_core_main_evidence_v0.1.0rc4.zip --summary core/snapshots/main_summary.json --aggregate core/snapshots/main_aggregate.json --output core_recomputed.json
```

This checks 135 runs and 270 aggregate cell-metric values. The original source-to-archive build was verified against its frozen training manifest; maximum absolute metric difference was below `1.5e-13`. The analysis is a reproducibility calculation, not a new statistical test.

To regenerate the full nine-condition Core dataset, run from this directory (requiring about 3.2 GB of output space):

```sh
python -m src.grammars.generator --config configs/experiments/synthetic_main_v1.json --output /new/empty/path
python -m scripts.verify_core_reconstruction --dataset /new/empty/path --expected core/dataset_checksums.json
```

The second command checks all 40 data/validation files against the frozen original manifest. The generation command refuses to overwrite an existing directory. It was independently run from the public resource package in a clean Python environment; the reproduced data passed the original file hashes.

## Evaluate a new prediction file

Each prediction row must have `record_id` and exactly one probability column: `probability`, `p_query_agent` or `p_first_agent`. CSV or JSONL is accepted; probabilities must be finite and in `[0,1]`. IDs must match the chosen dataset exactly. Example:

```sh
rolecue example-predictions --dataset binding/data/ro075_rm075/test.jsonl --output predictions.csv
rolecue evaluate --dataset binding/data/ro075_rm075/test.jsonl --predictions predictions.csv --output evaluation.json
rolecue example-predictions --dataset core/example_test.jsonl --output core_predictions.csv
rolecue evaluate --dataset core/example_test.jsonl --predictions core_predictions.csv
```

The first example writes the full-information oracle probability as a valid demonstration file; substitute predictions from any new model in the same format. The evaluator returns accuracy, NLL, Brier score, oracle NLL regret and posterior probability MAE, plus Binding version/distance/depth cells or Core cue-conflict strata. Metadata/oracle fields are never model inputs. `src/grammars/linearizer.py::model_example` defines the supervised token/label projection.

`examples/predict_constant.py` is a minimal adapter that reads only `record_id` and `tokens`; replace its probability function with any model. `DATA_CARD.md` defines fields, split construction, parameters, licensing and known limits.

## Rebuild, splits and scope

The included `core/prior_inventory/events.jsonl` has the exact SHA-256 expected by the frozen config; the config's path is relative to this directory. Binding data are fully included and need no regeneration, although the generator remains in `src/binding/build.py`.

Binding uses 1,600/200/200/200 disjoint semantic events per condition for train/dev/test/OOD. Reverse-role events, query targets, task versions and structural variants are kept together by unordered entity-pair/predicate group. `binding/data/information_certificate.json` gives same-suffix pairs that require more than final-suffix evidence, and the exact baselines. The frozen protocol and limits are in `docs/binding_protocol_v1_20261015.md`. The formal training configuration is `configs/experiments/train_binding_formal_v1.json`; `binding/results/summary.json` and each `runs/<condition>/<architecture>/seed<id>/` directory supply results and predictions. Trained `.pt` checkpoints are retained in the research workspace, not included in this compact candidate; the configuration and all released predictions support independent scoring and aggregate checks. The `binding/analysis/analysis.json` file separates seed and event-group uncertainty.

The smaller Core example selects 100 complete held-out semantic groups from the frozen `.75/.75` test set, retaining both role orientations. The full Core dataset is byte-reconstructible from the configuration and prior inventory. Original Core training checkpoints are not in this candidate. The separate evidence ZIP enables recomputation of all main-run scoring endpoints from released per-example probabilities, while the public example checks the evaluation interface. The candidate excludes UD text and third-party LaTeX styles. No expert linguistic validation or natural-language model evaluation is claimed.
