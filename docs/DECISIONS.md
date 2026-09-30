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
| D20 | Model + calibration | LightGBM (fixed baseline params, early stopping on val log-loss, 912 trees, `deterministic=true`) → isotonic calibration fitted on val, via a small custom wrapper | Deterministic training → byte-identical model on re-run, so DVC doesn't see phantom changes. Calibration cut test ECE from 0.016 to 0.005 with no AUC loss. Custom wrapper because sklearn's `CalibratedClassifierCV` validation can strip the pandas category dtype LightGBM needs. |
| D21 | Decision policy | Global threshold **0.24** (cost-minimizing on val). LGD 0.511 and haircut 0.807 estimated dollar-weighted on train. Frozen in `models/policy.json` | Matches the EDA break-even estimate (≈23%). Val favoured the global threshold over the per-loan expected-value rule (6.9% vs 5.3% savings), so D5 stands. Open question below. |
| D22 | Evaluation protocol | Test read once, by `evaluate` only. Baseline = approve-all (= LC's actual decisions). Also reports a hindsight-best threshold (never used for decisions) | Honest out-of-time estimate. The hindsight row shows how much the val-chosen threshold leaves on the table under drift. |
| D23 | DVC vs MLflow roles | DVC builds the model (pure, offline, reproducible). A separate `log_run` step records it in MLflow with lineage tags (git commit, DVC data + model md5s) and registers it. It refuses if `dvc status` isn't clean or git is dirty (unless `--allow-dirty`, which tags the version as unpromotable) | Training never depends on a network service. Every registered version traces to one commit and exact data hashes, answering "what data and code produced the model in production?" |
| D24 | Registry + promotion | Registered as an MLflow pyfunc (model.joblib + policy.json + riskflux code snapshot + pinned requirements). New versions get `@challenger`; `promote` scores challenger and `@champion` on the same labeled loans with the same economics and moves `@champion` only if cost drops ≥ 0.1%, ROC-AUC drops ≤ 0.005 and ECE ≤ 0.02. Every decision is tagged on the version | Aliases, not the deprecated stages. Promotion is decided by dollars, with guardrails on ranking and calibration. Serving follows `@champion`, so rollback = move the alias back. |
| D25 | API design | FastAPI: `/health` (liveness), `/ready` (readiness, 503 until the model loads), `/metadata`, `/predict`, `/predict/batch` (≤1,000). Request = `LoanApplication` (D18); response = calibrated PD, decision, expected loss, top-3 risk factors, policy flags, model version | Separate liveness and readiness so a missing model stops traffic without a restart loop. The service runs the exact training pipeline object, and an HTTP-level test proves identical scores to the batch path on 500 real loans. |
| D26 | Decisions + reason codes | `approve`/`reject` at the policy threshold; **`refer`** (human review) when the applicant is outside LC's approval box (FICO < 660, credit history < 36 months, loan/income > 0.5). Reason codes = top positive TreeSHAP contributions from LightGBM's built-in `pred_contrib`, with plain-English labels; missing inputs are labelled "(not provided)" | The model never saw applicants outside the box, so it shouldn't auto-decide them. TreeSHAP gives adverse-action reasons without the `shap` dependency; isotonic calibration is monotone, so the raw-score ranking stays valid. |
| D27 | Container | `fetch_model` pulls `@champion` from MLflow outside the build and verifies its md5 against the registry lineage tag; multi-stage uv Dockerfile (deps layer cached on `uv.lock`), slim runtime + `libgomp1`, non-root user, allowlist `.dockerignore`, one uvicorn worker per container | No registry credentials in the image; the image provably carries the registered model. The allowlist keeps `.env`, data and git history out of the build context by construction. Cloud Run scales by containers, so one worker each keeps memory predictable. |
| D28 | Simulated production traffic | `replay_data` DVC stage: 36-month individual loans issued 2017-05 … 2020-09 (after the modeling window), parsed by the same `clean()` with in-flight loans kept (label `<NA>`), 2,000 per month with a fixed seed (81,718 loans). Replayed in issue-date order through the running Docker image; the API's own prediction log is the monitoring input | Monitoring reads what the service actually logged, as in production. Grouping by application month (event time), not log time, keeps a fast replay equivalent to 41 real months. |
| D29 | Drift detection | Evidently, **PSI** per model input *after* feature engineering + PSI of predicted PD, vs a 20k-loan sample of the test period scored by the champion; plus missing-rate change (PSI ignores missing values) and decision rates. Monthly alert if ≥ 30% of inputs drift, prediction PSI ≥ 0.2, or any missing rate moves ≥ 10 points | PSI is the credit-risk industry standard. Engineered features = exactly what the model sees, and raw dates can't fake drift. The reference is where performance was actually measured. |

## Results (test: 239,705 loans issued 2016-07 … 2017-04)
| Metric | Value |
|---|---|
| ROC-AUC / PR-AUC | 0.688 / 0.274 |
| Brier / ECE | 0.122 / 0.005 |
| Cost vs approve-all | **−$7.08M (−2.9%)** at threshold 0.24, rejecting 13.9% of loans, catching 28% of defaults |
| Hindsight-best threshold | 0.285 → −3.6% (val-chosen threshold slightly too strict under drift) |
| Ablation (ROC-AUC) | LC `int_rate` alone 0.663 · our model 0.689 · our model + LC grade 0.697 |

## Drift findings (replay 2017-05 … 2020-09, `reports/drift/`)
- **2017-05 → 2018-01: stable.** Prediction PSI < 0.08, no alerts.
- **2018 onward: upstream data change.** `mths_since_last_record` PSI ≈ 0.9, `tax_liens` 0.25, `pub_rec` ~0.2, and public-record missing rates up 10+ points. This matches the credit bureaus removing tax liens and civil judgments from reports (NCAP, 2017-07 … 2018-04): a change in the data source, not in borrowers.
- **2020-04: COVID.** Prediction PSI jumps 0.04 → 0.42; the policy's reject rate falls from ~10% to 3% as Lending Club tightened credit (only much safer applicants were approved). Refer rate: 1.4% of replayed loans fell outside the training approval box, versus 0% in the test period.

## Open questions
- **Global threshold vs per-loan expected-value rule:** val favours the threshold (6.9% vs 5.3%), test favours the EV rule (3.4% vs 2.9%). Val is biased toward the threshold (it was tuned there; the EV rule has nothing to tune). Settle with a rolling-origin backtest over several periods, not by switching on test results.

## Known corners cut
- **No reject inference:** trained only on approved loans (selection bias).
- **Simulated label arrival:** the OOT split assumes labels exist immediately; real credit labels arrive months to years later.
- **Simplified economics:** fixed LGD; ignores interest collected before default, cost of funds, servicing fees, capital charges.
- **Bureau wave 3 unused:** 14 informative fields dropped because they only exist from 2015-12.
- **Bureau fields assumed as-of application** (per LC data dictionary) — not verifiable from the data.
- **Library versions aren't DVC deps:** `uv.lock` pins the environment per commit, but upgrading e.g. pandas won't trigger `dvc repro` on its own. Adding it as a dep would re-run everything on any package change.
- **Coarse code deps:** stages depend on whole files (e.g. `schemas.py`, `config.py`), so editing an unrelated part of a shared file re-runs stages that didn't strictly need it. Correct but occasionally wasteful (~1 min).
- **No hyperparameter tuning:** fixed baseline LightGBM settings. D3's expanding-window CV tuning is deferred; the infrastructure, not the last AUC point, is the goal.
- **Val does triple duty:** early stopping, calibration and threshold choice all use the same val split, making val metrics optimistic (val savings 6.9% vs test 2.9%). A real team would use a separate calibration/policy period.
- **Val-to-test gap from drift:** savings fall from 6.9% (val) to 2.9% (test); the threshold is fixed while the population shifts. This is what monitoring (Step 9) and retraining (Step 10) exist for.
- **Promotion uses the test split:** once several versions compete, test becomes a selection set. Step 10's retraining promotes on the newest labeled window instead.
- **Cross-version scoring in one process:** `promote` loads old and new versions in the same Python process, which assumes the old pickle still loads with current code. A real team would score each version in its own environment (e.g. its own image).
- **No approval workflow:** promotion is automatic once the rules pass. Banks usually require human sign-off (model risk management) before production.
- **API has no authentication or rate limiting:** fine for a demo; production needs auth (e.g. IAM / API gateway), rate limits and TLS termination.
- **Prediction logs contain applicant financial data:** a real lender would restrict access and set retention rules; here they go to stdout / a local JSONL file.
- **Policy box is hard-coded** from the training data's observed limits, not learned or versioned with the model.
- **Base images pinned by tag, not digest:** `python:3.11-slim-bookworm` can change underneath us; pinning the digest would make builds fully reproducible.
- **Expected loss uses the requested amount** as funded amount (unknown before funding).
- **Drift thresholds are heuristics** (PSI 0.1/0.2, 30% of features, 10-point missing change), not tuned to business impact. With 62 inputs, a few drift by chance each month; the share threshold absorbs that.
- **Drift ≠ performance loss:** PSI says inputs changed, not that decisions got worse. Realized performance needs labels, which arrive months later (Step 10 simulates their arrival).
- **Prediction log is a local JSONL file** (via a Docker volume). On Cloud Run the same records go to Cloud Logging; a real setup would sink them to a warehouse (e.g. BigQuery) for the monitoring job.
- **Monthly batch monitoring only:** no real-time alerting or dashboards.
- **No feature store:** a shared Python module replaces Feast/Tecton.
- **No shadow or canary rollout:** promotion goes straight to production.
- **Limited fairness analysis:** no protected attributes in the data.
- **Model only knows LC's approval box:** training data has FICO ≥ 660, credit history ≥ 36 months, loan/income ≤ 0.5. Applications outside it are extrapolation — to be flagged by the API (Step 8).
- **Pickled pipeline is tied to the `riskflux` code version:** the model file references `riskflux.features` classes, so code and model must ship together (D10 bakes both into one image).
