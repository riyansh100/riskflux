# Design Decisions

Log of architectural decisions and the reasoning behind each.

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Dataset | Lending Club 2007–2020Q3 (Kaggle: ethon0426) | Real issuance dates enable genuine temporal drift; immature late vintages serve as a drift-replay stream. |
| D2 | Label + population | 36-month loans, matured cohorts only. Charged Off/Default = 1, Fully Paid = 0 | Unmatured vintages over-represent early defaults and early payoffs (right-censoring). 60-month loans excluded so dropping them over time can't create fake drift. |
| D3 | Validation | Out-of-time split by `issue_d` + expanding-window CV for tuning | Random K-fold leaks future information and inflates metrics. |
| D4 | LC grade / sub_grade / int_rate | Excluded from model features; included only in an ablation | These are outputs of Lending Club's own risk model; using them stacks on another model. `int_rate` is still used in the cost function. |
| D5 | Cost function | Loss = PD × loan_amnt × LGD (LGD estimated from training-period defaults). Missed profit = contractual interest × empirical prepayment haircut. Global threshold minimizes total $ cost | Optimizes business outcome, not accuracy. Haircut corrects for early payoff. |
| D6 | Class imbalance | No resampling; calibrate probabilities; cost-based threshold | Resampling distorts probabilities that expected-loss math depends on. ~20% default rate is mild imbalance. |
| D7 | Algorithm | LightGBM | Native categorical handling; fast on ~1M rows. |
| D8 | Tracking + data remote | DagsHub (hosted MLflow + DVC remote) | Free, publicly browsable runs; enables CI retraining. |
| D9 | Model promotion | MLflow registry aliases (`@challenger` → `@champion`) | Registry stages are deprecated since MLflow 2.9. |
| D10 | Model packaging | Model baked into Docker image at build time | Image tag = model version; rollback = redeploy old image; no runtime MLflow dependency. |
| D11 | Deployment | GCP Cloud Run | Runs containers directly, scales to zero, free tier. |
| D12 | Additions | pandera data validation; SHAP reason codes in API | Catch bad data before training; US lenders must give reasons for denials (adverse action). |
| D13 | Tooling | Python 3.11, uv, ruff, pytest | Fast, reproducible (lockfile), standard. |

## Known corners cut
- **No reject inference:** trained only on approved loans (selection bias).
- **Simulated label arrival:** real credit labels arrive months to years later.
- **Simplified economics:** fixed LGD; no cost of funds, servicing fees, or capital charges.
- **No feature store:** a shared Python module replaces Feast/Tecton.
- **No shadow or canary rollout:** promotion goes straight to production.
- **Limited fairness analysis:** no protected attributes in the data.