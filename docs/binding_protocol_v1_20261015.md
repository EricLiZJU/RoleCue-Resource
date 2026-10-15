# RoleCue-Binding: frozen protocol and result audit

Status: 45/45 formal runs complete. This extension was designed after the older E3 experiment exposed a locally sufficient suffix. It is a new formal task, not a preregistered interpretation of E3. The new task data, metrics and hyperparameters were frozen after development checks and before formal test scoring.

## Construction and information test

The stipulated language has one delimited transitive main clause and zero or more legal, separately delimited distractor clauses. The main clause has two distinct candidates, complementary `<NOM>`/`<ACC>` tokens and a predicate. Distractors use other entities and their own complementary markers. A query names one of the two main candidates, balanced over candidate positions. The label asks whether that **named** entity is the main-clause Agent. `src/binding/task.py::parse` rejects malformed delimiters, duplicate marker types, reused distractor entities and a query outside the main candidates. It derives the same first-candidate/posterior calculation as the independent two-state enumerator for every generated record (tolerance `1e-12`).

Each event has two equal-length versions. In `local`, a final query-position flag and main marker-direction summary permit a last-five-token solution. In `binding`, both flags are masked; the last nine tokens comprise only a distractor tail, flags/masks and the query. The full visible sentence remains sufficient. `information_certificate.json` contains explicit identical-suffix pairs with opposite truths and different exact posteriors, and compares suffix, order-only, marker-only, entity-unbound and full-oracle observers. Held-out test witness counts by condition are 588 (`.55/.95`), 426 (`.75/.75`) and 570 (`.95/.55`). The full parser achieves exact analytic probability agreement. The suffix Binding observer has population NLL `ln(2)=0.6931`, whereas the local suffix observer equals the full oracle.

The evidence is an information-access proof within the stipulated generator. It does not establish typological naturalness or an expert linguistic validation. It does establish that any final-nine-token-only strategy must fail for at least one member of each certified pair. The main clause and REL grammar are defined in `LRE_paper/sections/06_binding.tex`; legal/invalid examples and baseline definitions appear in `LRE_paper/appendix_lre.tex`.

## Split, development and freeze

An unordered entity pair plus predicate is the split key. Reverse-role events, both query targets, both task versions and all distractor structures stay in the same split. There are 48 entity symbols and 8 predicate symbols, with 1,600 training, 200 development, 200 test and 200 OOD semantic events **per reliability condition**. The three conditions are order/marker reliability `.55/.95`, `.75/.75` and `.95/.55`. Training uses `(distance, depth)=(1,0),(2,0),(2,1)`; test repeats these cells on held-out event groups, allowing the `(2,0)` vs `(2,1)` equal-length structural contrast. The `(4,2)` OOD cell changes length and depth together and is interpreted only as a joint extrapolation. No unseen-vocabulary or isolated composition-rule claim is made.

Smoke checks used 3 architectures, 1 seed and 2 epochs to verify training/reload. A separate development-only one-seed, eight-epoch check established that the matched-capacity models learned the local control. Neither contributes formal test results. The frozen formal config is `configs/experiments/train_binding_formal_v1.json`, SHA-256 `ad1cc23992552d81af12eeb5075ce75170dc97f50a763b29bf4cd2ac523dbcb4`. Data manifest SHA-256 is `8f6743082665aad06659d8ef1c5be331140f77f278db9bbc7aeadf487100cada`. This run uses RNN/LSTM/Transformer, five seeds (11–15), eight epochs, shared mixed-version training, development-NLL checkpoint selection and approximate 71k parameter budgets. Seed IDs identify paired **datasets/analysis cells**, not identical initializations across architectures.

## Metrics and uncertainty

Prespecified primary endpoints are Binding test NLL regret relative to the probability oracle, posterior MAE, the local/Binding contrast and the suffix/full-oracle information gap. Structural and OOD cuts are secondary. The report computes each seed separately and a five-seed Student-t interval for training variation. A separate 1,000-replicate event-group bootstrap resamples unordered pair/predicate groups while fixing trained runs; it estimates held-out event sampling variation conditional on the five fitted models. These intervals are not pooled. Test surface noise is fixed in this new dataset and has no independent replication interval. As with the Core follow-up, inferential contrasts added after inspection are labeled exploratory. Negative finite-test NLL regret is possible because the oracle NLL is computed on realized labels, not an expected-loss integral.

The 45 run IDs, checkpoint selection, reload checks and score files live under `results/runs/binding_formal_v1_20261015`; aggregate results and per-seed uncertainty live under `results/aggregate/binding_formal_v1`. All formal results come from held-out event groups. No architecture-equivalence inference is made from a nonsignificant contrast.

## Principal observed results

| Reliability `r_o/r_m` | Architecture | Binding accuracy | Binding NLL regret (nats) | Local NLL regret (nats) |
|---|---|---:|---:|---:|
| .55/.95 | RNN | .501 | .554 | .003 |
| .55/.95 | LSTM | .500 | .553 | -.002 |
| .55/.95 | Transformer | .508 | .552 | -.000 |
| .75/.75 | RNN | .501 | .202 | .002 |
| .75/.75 | LSTM | .500 | .201 | .001 |
| .75/.75 | Transformer | .523 | .193 | .003 |
| .95/.55 | RNN | .499 | .478 | -.001 |
| .95/.55 | LSTM | .500 | .478 | .000 |
| .95/.55 | Transformer | .942 | .012 | .001 |

The strong-order Transformer result is consistent with associating the query name with a distant candidate **position** and exploiting the reliable order cue. It does not demonstrate use of the weak marker. Its Binding regret five-seed 95% interval is `[0.00025, 0.02396]` nats; the conditional event-group bootstrap interval is `[-0.00295, 0.02635]`. The `(4,2)` cut lowers mean Binding accuracy to `.915` and increases posterior MAE to `6.45` percentage points, against about `3.0` points in either length-matched `(2,0)/(2,1)` test cell. The other model groups near chance on Binding still learned the local control; this shows the diagnostic discriminates information availability from model use of available information. This is a finding about these models, budgets, training distribution and symbolic grammar.
