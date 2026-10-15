# Formal RoleCue-Binding analysis

45/45 runs; data manifest SHA-256: `8f6743082665aad06659d8ef1c5be331140f77f278db9bbc7aeadf487100cada`.

Five-seed intervals use trained runs. Group-bootstrap intervals resample unordered entity-pair/predicate groups while holding these five runs fixed; the two are not combined.

| Condition | Model | Binding accuracy | Binding NLL regret | Local NLL regret | Binding minus local regret | Binding posterior MAE |
|---|---|---:|---:|---:|---:|---:|
| ro055_rm095 | rnn | 0.501 | 0.554 | 0.003 | 0.550 | 0.449 |
| ro055_rm095 | lstm | 0.500 | 0.553 | -0.002 | 0.555 | 0.449 |
| ro055_rm095 | transformer | 0.508 | 0.552 | -0.000 | 0.552 | 0.447 |
| ro075_rm075 | rnn | 0.501 | 0.202 | 0.002 | 0.200 | 0.242 |
| ro075_rm075 | lstm | 0.500 | 0.201 | 0.001 | 0.200 | 0.238 |
| ro075_rm075 | transformer | 0.523 | 0.193 | 0.003 | 0.190 | 0.234 |
| ro095_rm055 | rnn | 0.499 | 0.478 | -0.001 | 0.479 | 0.450 |
| ro095_rm055 | lstm | 0.500 | 0.478 | 0.000 | 0.477 | 0.450 |
| ro095_rm055 | transformer | 0.942 | 0.012 | 0.001 | 0.011 | 0.031 |

The complete JSON contains each seed, both uncertainty sources, equal-length cells, OOD cells, and suffix-collision certificates.
No architecture equivalence or inferential mechanism claim is made. The d4/h2 cut changes length and depth jointly.
