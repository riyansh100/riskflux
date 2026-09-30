"""Drift job: compare each month of served traffic to the reference period.

Reference = the test period (2016-07..2017-04), where the champion's performance and
dollar savings were measured, scored by the champion. Current = what the API logged,
grouped by application month (event time, not log time).

What is compared (per month):
  - the 62 model inputs AFTER feature engineering (raw dates would drift trivially),
    with PSI via Evidently -> share of drifted features, top drifted features
  - the predicted-PD distribution (prediction drift, PSI)
  - missing-value rates (PSI ignores missing values, so a feature going blank
    would otherwise go unnoticed)
  - decision rates (reject / refer)

A month raises an alert if any rule in params.yaml (monitoring) is breached. The latest
month's verdict is what the retraining workflow (Step 10) acts on.

Run: uv run python -m riskflux.monitoring.drift [--log data/monitoring/predictions.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from riskflux import columns as C
from riskflux.config import (
    DRIFT_REPORT_HTML,
    DRIFT_SUMMARY,
    DRIFT_TIMELINE_PNG,
    PREDICTION_LOG,
    MonitoringParams,
    load_params,
)
from riskflux.features.engineer import MODEL_CATEGORICAL, MODEL_FEATURES, MODEL_NUMERIC, engineer
from riskflux.models.io import load_model, load_split, write_json
from riskflux.schemas import LoanApplication, applications_to_frame

PD = "pd"
TOP_N = 5


def load_prediction_log(path: Path) -> pd.DataFrame:
    """Prediction log -> typed frame via the serving contract (same parsing as the API)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    applications = [LoanApplication.model_validate(r["features"]) for r in records]
    frame = applications_to_frame(applications)
    frame[PD] = [r["pd"] for r in records]
    frame["decision"] = [r["decision"] for r in records]
    frame["model_version"] = [r["model_version"] for r in records]
    return frame


def monitored(frame: pd.DataFrame, pd_hat: pd.Series) -> pd.DataFrame:
    """Exactly what the model sees (engineered features) plus its output."""
    out = engineer(frame).reset_index(drop=True)
    out[PD] = np.asarray(pd_hat, dtype=float)
    return out


def reference_frame(n: int, seed: int) -> pd.DataFrame:
    pipeline, policy = load_model()
    test = load_split("test").sample(n=n, random_state=seed)
    ref = monitored(test, pipeline.predict_proba(test)[:, 1])
    ref["decision"] = np.where(ref[PD] >= policy["threshold"], "reject", "approve")
    return ref


def _definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=[*MODEL_NUMERIC, PD], categorical_columns=MODEL_CATEGORICAL
    )


def drift_report(current: pd.DataFrame, reference: pd.DataFrame, params: MonitoringParams):
    report = Report(
        [
            DataDriftPreset(columns=MODEL_FEATURES, method="psi", threshold=params.psi_threshold),
            ValueDrift(column=PD, method="psi", threshold=params.psi_threshold),
        ]
    )
    columns = [*MODEL_FEATURES, PD]
    return report.run(
        Dataset.from_pandas(current[columns], data_definition=_definition()),
        Dataset.from_pandas(reference[columns], data_definition=_definition()),
    )


def summarize_window(
    current: pd.DataFrame, reference: pd.DataFrame, params: MonitoringParams, snapshot=None
) -> dict:
    snapshot = snapshot or drift_report(current, reference, params)
    feature_psi, prediction_psi = {}, None
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("ValueDrift"):
            if config["column"] == PD:
                prediction_psi = float(metric["value"])
            else:
                feature_psi[config["column"]] = float(metric["value"])

    drifted = {k: v for k, v in feature_psi.items() if v >= params.psi_threshold}
    missing_change = (
        current[MODEL_FEATURES].isna().mean() - reference[MODEL_FEATURES].isna().mean()
    ).abs()
    top_missing = missing_change.sort_values(ascending=False).head(1)
    summary = {
        "loans": len(current),
        "drifted_share": round(len(drifted) / len(MODEL_FEATURES), 4),
        "top_drifted": {
            k: round(v, 4) for k, v in sorted(drifted.items(), key=lambda kv: -kv[1])[:TOP_N]
        },
        "prediction_psi": round(prediction_psi, 4),
        "mean_pd": round(float(current[PD].mean()), 4),
        "reject_rate": round(float((current["decision"] == "reject").mean()), 4),
        "refer_rate": round(float((current["decision"] == "refer").mean()), 4),
        "max_missing_change": {top_missing.index[0]: round(float(top_missing.iloc[0]), 4)},
    }
    summary["alerts"] = alerts(summary, params)
    return summary


def alerts(window: dict, params: MonitoringParams) -> list[str]:
    raised = []
    if window["drifted_share"] >= params.drifted_share_alert:
        raised.append(
            f"{window['drifted_share']:.0%} of model inputs drifted (PSI >= {params.psi_threshold})"
        )
    if window["prediction_psi"] >= params.prediction_psi_alert:
        raised.append(
            f"prediction PSI {window['prediction_psi']:.3f} >= {params.prediction_psi_alert}"
        )
    ((feature, change),) = window["max_missing_change"].items()
    if change >= params.missing_share_alert:
        raised.append(f"missing rate of {feature} moved {change:.0%}")
    return raised


def plot_timeline(summary: dict, params: MonitoringParams, path: Path) -> None:
    """Two stacked panels over replay months: prediction PSI and reject rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = pd.PeriodIndex(list(summary["months"]), freq="M").to_timestamp()
    windows = list(summary["months"].values())
    blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    top.plot(months, [w["prediction_psi"] for w in windows], color=blue, linewidth=2)
    top.axhline(params.prediction_psi_alert, color=muted, linestyle="--", linewidth=1)
    top.set_ylabel("prediction PSI")
    top.set_title("Drift of the model's predicted PD vs the test period", loc="left")
    reference_reject = summary["reference"]["reject_rate"] * 100
    bottom.plot(months, [w["reject_rate"] * 100 for w in windows], color=orange, linewidth=2)
    bottom.axhline(reference_reject, color=muted, linestyle="--", linewidth=1)
    bottom.set_ylabel("reject rate (%)")
    bottom.set_title("Share of applications the policy rejects", loc="left")
    for ax in (top, bottom):
        ax.axvspan(
            pd.Timestamp("2017-07-01"), pd.Timestamp("2018-04-01"), color="#e6e5e0", zorder=0
        )
        ax.axvline(pd.Timestamp("2020-03-01"), color=muted, linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    top.annotate(
        f"alert threshold {params.prediction_psi_alert}",
        (months[0], params.prediction_psi_alert),
        xytext=(0, 4),
        textcoords="offset points",
        color=muted,
        fontsize=8,
    )
    top.annotate(
        "bureaus drop tax liens\n& judgments (NCAP)",
        (pd.Timestamp("2017-07-15"), 0.30),
        color=muted,
        fontsize=8,
    )
    top.annotate(
        "COVID-19", (pd.Timestamp("2020-02-10"), 0.30), color=muted, fontsize=8, ha="right"
    )
    bottom.annotate(
        f"reference {reference_reject:.1f}%",
        (months[0], reference_reject),
        xytext=(0, 4),
        textcoords="offset points",
        color=muted,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=PREDICTION_LOG)
    args = parser.parse_args()
    params = load_params().monitoring

    reference = reference_frame(params.reference_sample, params.seed)
    served = load_prediction_log(args.log)
    current = monitored(served, served[PD])
    current["decision"] = served["decision"].to_numpy()
    months = served[C.ISSUE_DATE].dt.to_period("M").to_numpy()

    results, snapshot = {}, None
    for month in sorted(set(months)):
        window = current[months == month]
        snapshot = drift_report(window, reference, params)
        results[str(month)] = summarize_window(window, reference, params, snapshot)
        w = results[str(month)]
        print(
            f"{month}  drifted={w['drifted_share']:.0%}  pred_psi={w['prediction_psi']:.3f}  "
            f"reject={w['reject_rate']:.1%}  {'ALERT' if w['alerts'] else 'ok'}"
        )

    latest = max(results)
    summary = {
        "reference": {
            "period": "test split (2016-07..2017-04)",
            "loans": len(reference),
            "mean_pd": round(float(reference[PD].mean()), 4),
            "reject_rate": round(float((reference["decision"] == "reject").mean()), 4),
        },
        "model_versions_seen": sorted(set(served["model_version"])),
        "latest_month": latest,
        "drift_detected": bool(results[latest]["alerts"]),
        "months": results,
    }
    write_json(DRIFT_SUMMARY, summary)
    snapshot.save_html(str(DRIFT_REPORT_HTML))
    plot_timeline(summary, params, DRIFT_TIMELINE_PNG)
    print(
        f"latest month {latest}: {'DRIFT DETECTED' if summary['drift_detected'] else 'no drift'} "
        f"-> {DRIFT_SUMMARY}, {DRIFT_REPORT_HTML}"
    )


if __name__ == "__main__":
    main()
