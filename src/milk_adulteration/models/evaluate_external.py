"""Score the trained cascade on a CSV of samples, e.g. real lab samples.

    python -m milk_adulteration.models.evaluate_external real_samples.csv \\
        --label is_adulterated --adulterant adulterant \\
        --rename "Fat=fat_pct,SNF=snf_pct" --out reports/external.md \\
        --predictions reports/external_predictions.csv

`--predictions` writes one row per sample with the model's answer (works without
labels too). `--freezing-point-unit`, `--conductivity-unit` and `--density-unit`
convert common instrument units to the ones the model expects.

The CSV needs the six readings (renamed with --rename if its headers differ) and,
for accuracy numbers, a 0/1 label column. An adulterant-name column is optional;
names are mapped to the model's classes (water, starch, urea, detergent,
sodium_bicarbonate, glucose, skimmed_milk_powder, other).

Nothing here trains or tunes on the file, so it is a fair held-out check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milk_adulteration.config import (
    ADULTERANT_CLASSES,
    FEATURES,
    OTHER_CLASS,
    PURE_CLASS,
    STAGE2_CLASSES,
    load_params,
    resolve,
)
from milk_adulteration.data.validate import split_valid_readings
from milk_adulteration.evaluation import binary_metrics, bootstrap_ci, multiclass_metrics
from milk_adulteration.models.cascade import Cascade

PURE_NAMES = {"", "none", "pure", "no", "nan", "0"}


def normalise_adulterant(value: Any) -> str:
    """Map free-text adulterant names onto model classes; unknown names become `other`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return PURE_CLASS
    text = str(value).strip()
    if text.lower() in PURE_NAMES:
        return PURE_CLASS
    by_lower = {k.lower(): v for k, v in ADULTERANT_CLASSES.items()}
    by_lower.update({c: c for c in STAGE2_CLASSES})
    key = text.lower().replace("-", " ")
    return by_lower.get(key, by_lower.get(key.replace(" ", "_"), OTHER_CLASS))


# ISO 5764 / IDF 108: freezing point in degrees Celsius from degrees Hortvet.
HORTVET_TO_CELSIUS = 0.9658


def apply_units(
    df: pd.DataFrame,
    freezing_point: str = "C",
    conductivity: str = "mS/cm",
    density: str = "g/mL",
) -> pd.DataFrame:
    """Convert readings to the model's units: °C, mS/cm and g/mL."""
    df = df.copy()
    if freezing_point.upper() == "H":
        df["freezing_point"] = pd.to_numeric(df["freezing_point"], errors="coerce") * (
            HORTVET_TO_CELSIUS
        )
    if conductivity.lower() in ("us/cm", "µs/cm"):
        df["conductivity"] = pd.to_numeric(df["conductivity"], errors="coerce") / 1000
    if density.upper() == "CLR":  # corrected lactometer reading, e.g. 28 -> 1.028
        df["density"] = 1 + pd.to_numeric(df["density"], errors="coerce") / 1000
    return df


def predict_rows(model: Cascade, df: pd.DataFrame) -> pd.DataFrame:
    """One output row per input row, in input order: the inputs unchanged (lab
    labels included), then `status` (scored or rejected), the model's answer in
    `pred_*` columns and any `reject_reason`."""
    df = df.reset_index(drop=True)
    valid, rejected = split_valid_readings(df)
    pred = model.predict(valid) if len(valid) else pd.DataFrame(index=valid.index)
    out = df.copy()
    out["status"] = np.where(out.index.isin(valid.index), "scored", "rejected")
    for col in ("is_adulterated", "risk_score", "band", "adulterant", "confidence"):
        out[f"pred_{col}"] = pred[col] if col in pred else None
    out["pred_unusual_readings"] = (
        pred["unusual_readings"].map(", ".join) if "unusual_readings" in pred else None
    )
    out["reject_reason"] = rejected["reject_reason"] if len(rejected) else None
    return out


def parse_rename(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    pairs = [p.split("=", 1) for p in spec.split(",") if p.strip()]
    return {src.strip(): dst.strip() for src, dst in pairs}


def evaluate_external(
    model: Cascade,
    df: pd.DataFrame,
    label: str | None,
    adulterant: str | None,
    rounds: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    valid, rejected = split_valid_readings(df.reset_index(drop=True))
    out: dict[str, Any] = {
        "rows": len(df),
        "scored": len(valid),
        "rejected": len(rejected),
        "reject_reasons": rejected["reject_reason"].value_counts().head(10).to_dict(),
    }
    if valid.empty:
        return out
    pred = model.predict(valid)
    raw = model.raw_score(valid)
    out["flag_rate"] = float(pred["is_adulterated"].mean())
    out["bands"] = pred["band"].value_counts().to_dict()
    out["predicted_adulterants"] = pred["adulterant"].dropna().value_counts().to_dict()

    if adulterant and adulterant in valid:
        cls = valid[adulterant].map(normalise_adulterant)
    else:
        cls = None
    if label and label in valid:
        y = pd.to_numeric(valid[label], errors="coerce")
        if y.isna().any() or not set(y.unique()) <= {0, 1}:
            raise ValueError(f"label column {label!r} must contain only 0 and 1")
        y = y.astype(int).to_numpy()
    elif cls is not None:
        y = (cls != PURE_CLASS).astype(int).to_numpy()
    else:
        return out  # no labels: prediction summary only

    out["positives"] = int(y.sum())
    views = {"all": np.ones(len(y), dtype=bool)}
    if cls is not None:
        views["detectable"] = (cls != OTHER_CLASS).to_numpy()
    out["stage1"] = {}
    for view, keep in views.items():
        yk, sk = y[keep], raw[keep]
        if yk.min() == yk.max():
            out["stage1"][view] = {"note": "only one class present; no recall/precision"}
            continue
        m = binary_metrics(yk, sk, model.threshold)
        m["ci95"] = bootstrap_ci(yk, sk, model.threshold, rounds, seed)
        out["stage1"][view] = m

    if cls is not None and (cls != PURE_CLASS).any():
        pos = cls != PURE_CLASS
        names = np.asarray(model.classes)[model.adulterant_proba(valid[pos]).argmax(axis=1)]
        out["stage2"] = multiclass_metrics(cls[pos], names, STAGE2_CLASSES)
        present = sorted(set(cls[pos]))
        out["stage2"]["classes_present"] = present
        out["stage2"]["macro_f1_present"] = multiclass_metrics(cls[pos], names, present)["macro_f1"]
    return out


def render(r: dict[str, Any], source: str, model: Cascade) -> str:
    lines = [
        "# External evaluation",
        "",
        f"Source: `{source}`. Model: run `{model.metadata.get('mlflow_run_id', '?')}`, "
        f"threshold {model.threshold:.4f}.",
        "",
        f"- Rows: {r['rows']}, scored: {r['scored']}, rejected by validation: {r['rejected']}",
    ]
    for reason, n in r.get("reject_reasons", {}).items():
        lines.append(f"  - {n} × {reason}")
    if "flag_rate" in r:
        lines.append(f"- Flag rate: {r['flag_rate']:.1%}; bands: {r['bands']}")
    if "stage1" not in r:
        lines += ["", "No labels given, so no accuracy figures."]
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "## Stage 1: pure vs adulterated (95% bootstrap CI)",
        "",
        "| View | Recall | Precision | PR-AUC | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for view, m in r["stage1"].items():
        if "note" in m:
            lines.append(f"| {view} | {m['note']} | | | | | | |")
            continue
        ci = m["ci95"]
        lines.append(
            f"| {view} | {m['recall']:.3f} ({ci['recall'][0]:.2f}–{ci['recall'][1]:.2f}) | "
            f"{m['precision']:.3f} ({ci['precision'][0]:.2f}–{ci['precision'][1]:.2f}) | "
            f"{m['pr_auc']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    if "stage2" in r:
        s2 = r["stage2"]
        lines += [
            "",
            "## Stage 2: which adulterant (all adulterated rows)",
            "",
            f"Macro-F1 over classes present ({', '.join(s2['classes_present'])}): "
            f"**{s2['macro_f1_present']:.3f}**",
            "",
            "| Class | F1 |",
            "|---|---|",
            *(f"| {c} | {s2['per_class_f1'][c]:.2f} |" for c in s2["classes_present"]),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--label", help="0/1 column: 1 = adulterated")
    parser.add_argument("--adulterant", help="column naming the adulterant (blank/none = pure)")
    parser.add_argument("--rename", help="header mapping, e.g. 'Fat=fat_pct,SNF=snf_pct'")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/external.md"))
    parser.add_argument("--predictions", type=Path, help="write per-sample results to this CSV")
    parser.add_argument(
        "--freezing-point-unit", choices=["C", "H"], default="C", help="C = °C, H = °Hortvet"
    )
    parser.add_argument("--conductivity-unit", choices=["mS/cm", "uS/cm"], default="mS/cm")
    parser.add_argument(
        "--density-unit",
        choices=["g/mL", "CLR"],
        default="g/mL",
        help="CLR = corrected lactometer reading (28 means 1.028)",
    )
    args = parser.parse_args()

    params = load_params()
    model = Cascade.load(args.model or resolve(params["train"]["model"]))
    df = pd.read_csv(args.csv, dtype={"sample_id": str}).rename(columns=parse_rename(args.rename))
    missing = [c for c in FEATURES if c not in df]
    if missing:
        raise SystemExit(
            f"{args.csv} is missing columns {missing}. Found: {list(df.columns)}. "
            "Use --rename 'YourName=model_name,...' to map headers."
        )
    df = apply_units(df, args.freezing_point_unit, args.conductivity_unit, args.density_unit)
    r = evaluate_external(model, df, args.label, args.adulterant)
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        predict_rows(model, df).to_csv(args.predictions, index=False)
        print(f"Per-sample results: {args.predictions}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(r, str(args.csv), model))
    args.out.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
