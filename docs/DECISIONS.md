# Design Decisions

Log of architectural decisions and the reasoning behind each.
Evidence for data decisions: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Column-level classification: [`COLUMN_AUDIT.md`](COLUMN_AUDIT.md).

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Dataset | Lending Club 2007–2020Q3 (Kaggle: ethon0426) | Real issuance dates enable genuine temporal drift; immature late vintages serve as a drift-replay stream. |
| D2 | Label + population | 36-month, individual applications, issued **2013-01 … 2017-04**, resolved only (953,889 loans, 14.7% default). Charged Off/Default = 1, Fully Paid = 0 | ≥98% of loans issued ≤ 2017-04 are resolved; after that, right-censoring distorts the label. 2013-01 start = every loan has the bureau fields added in 2012. Joint apps are rare and carry their own field set. |
| D3 | Validation | Out-of-time split: train 2013-01…2015-12 (546k) · val 2016-01…2016-06 (168k) · test 2016-07…2017-04 (240k) | Random K-fold leaks future information and hides the drift visible in the data (default rate 12% → 16%). |
| D4 | LC decision columns | Excluded from features: `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt(_inv)`, `initial_list_status`, `verification_status`. Grade ablation only | Outputs of Lending Club's own underwriting. `installment` reconstructs `int_rate` (Spearman 0.9997). LC verifies applicants it already considers risky ("Verified" defaults 18% vs 11%). |
| D4b | Other exclusions | Post-origination (32 cols, leakage), bureau wave 3 (14 cols, only from 2015-12), joint-only (15), proxies (`zip_code`, `addr_state`, `emp_title`) | See `COLUMN_AUDIT.md`. 61 candidate features remain. |
| D5 | Cost function | Loss = PD × funded_amnt × LGD. Missed profit = contractual interest × prepayment haircut. Global threshold minimizes total $ cost. LGD & haircut estimated on train split only | EDA preview: LGD ≈ 0.52, haircut ≈ 0.80, break-even PD ≈ 23% for an average loan — far from the naive 0.5. |
| D6 | Class imbalance | No resampling; calibrate probabilities; cost-based threshold | Resampling distorts probabilities that expected-loss math depends on. 14.7% default rate is mild imbalance. |
| D7 | Algorithm | LightGBM | Native categorical handling; fast on ~1M rows. |
| D8 | Tracking + data remote | DagsHub (hosted MLflow + DVC remote) | Free, publicly browsable runs; enables CI retraining. |
| D9 | Model promotion | MLflow registry aliases (`@challenger` → `@champion`) | Registry stages are deprecated since MLflow 2.9. |
| D10 | Model packaging | Model baked into Docker image at build time | Image tag = model version; rollback = redeploy old image; no runtime MLflow dependency. |
| D11 | Deployment | GCP Cloud Run | Runs containers directly, scales to zero, free tier. |
| D12 | Additions | pandera data validation; SHAP reason codes in API | Catch bad data before training; US lenders must give reasons for denials (adverse action). |
| D13 | Tooling | Python 3.11, uv, ruff, pytest | Fast, reproducible (lockfile), standard. |
| D14 | Ingest | Stream CSV → parquet, every column as string | 1.77 GB CSV doesn't fit in pandas on 8 GB RAM; type inference from early rows is wrong for columns empty before 2012. Parsing is explicit in the clean stage. |
| D15 | Clean stage | Stateless, strict parsing, two outputs: `loans_clean` (id, issue_d, y, 61 features) and `loans_aux` (LC decisions, outcomes, fairness columns). pandera contract validated before writing | Nothing learned from data here, so no test-set leakage. Unexpected source values fail loudly. Leakage columns physically can't reach the model file; `strict=True` rejects extra columns. |
| D16 | Outliers | No capping/winsorizing of `annual_inc`, `revol_util` | Tree models split on thresholds, so extreme values don't distort them (supersedes the EDA note). Revisit only for a linear baseline or drift statistics. |
| D17 | Feature pipeline | sklearn `Pipeline`: stateless `FeatureEngineer` (FICO midpoint, credit-history months, loan/income, revol_bal/income) → `CategoricalVocab` (category list learned on train). 62 model inputs. No imputation or scaling | Training and serving call the same fitted object; the model is appended to it in Step 6. LightGBM handles NaN and is scale-invariant, so imputers/scalers would add state without changing predictions. Unseen categories → NaN, never a crash or a re-numbered code. `issue_d` is used to compute history length but is not a feature. |
| D18 | Serving contract | pydantic `LoanApplication` generated from `columns.py` + the same bounds as the pandera schema; `extra="forbid"` | One source of truth for batch and API rules. Clients sending `grade`, `int_rate`, etc. get a 422. Parity test proves identical features for the same loan via both paths (synthetic + 2,000 real loans). |
| D19 | DVC pipeline | `dvc.yaml` stages ingest → clean → split. Each stage lists every riskflux module it imports as a dep (enforced by a test) plus only the params sections it reads. All outputs cached and pushed to DagsHub; split sizes/default rates tracked as DVC metrics | A missing code dep = stale outputs with no warning, so a test proves the dep lists are complete. Outputs are pushed so any commit can be restored with `dvc pull` in ~20 s instead of recomputed. Verified: repro is a no-op when nothing changed, a split-param edit invalidates only `split`, and ingest output is byte-identical across runs. |

## Known corners cut
- **No reject inference:** trained only on approved loans (selection bias).
- **Simulated label arrival:** the OOT split assumes labels exist immediately; real credit labels arrive months to years later.
- **Simplified economics:** fixed LGD; ignores interest collected before default, cost of funds, servicing fees, capital charges.
- **Bureau wave 3 unused:** 14 informative fields dropped because they only exist from 2015-12.
- **Bureau fields assumed as-of application** (per LC data dictionary) — not verifiable from the data.
- **Library versions aren't DVC deps:** `uv.lock` pins the environment per commit, but upgrading e.g. pandas won't trigger `dvc repro` on its own. Adding it as a dep would re-run everything on any package change.
- **Coarse code deps:** stages depend on whole files (e.g. `schemas.py`, `config.py`), so editing an unrelated part of a shared file re-runs stages that didn't strictly need it. Correct but occasionally wasteful (~1 min).
- **No feature store:** a shared Python module replaces Feast/Tecton.
- **No shadow or canary rollout:** promotion goes straight to production.
- **Limited fairness analysis:** no protected attributes in the data.
- **Model only knows LC's approval box:** training data has FICO ≥ 660, credit history ≥ 36 months, loan/income ≤ 0.5. Applications outside it are extrapolation — to be flagged by the API (Step 8).
- **Pickled pipeline is tied to the `riskflux` code version:** the model file references `riskflux.features` classes, so code and model must ship together (D10 bakes both into one image).
