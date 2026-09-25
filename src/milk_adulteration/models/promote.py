"""DVC stage `promote`: register the cascade in MLflow and promote it if it earns it.

A candidate becomes the `production` alias only if, on the test set's detectable
adulterants, it
1. meets the hard gates (recall and precision targets), and
2. is no worse than the current production model on both PR-AUC and recall.

Ties promote: on this synthetic data both metrics sit at 1.0, and a retrain on new
data or code should still be able to replace the old model.

Writes `models/model_info.json` (read by the API) and `reports/promotion.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from milk_adulteration.config import load_params, resolve, tracking_uri
from milk_adulteration.models.cascade import Cascade

log = logging.getLogger(__name__)

COMPARED = ("pr_auc", "recall")


class CascadePyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper so the registry holds a loadable model."""

    def load_context(self, context):
        self.model = Cascade.load(context.artifacts["cascade"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.model.predict(model_input)


def decide(
    candidate: dict[str, float], current: dict[str, float] | None, p: dict[str, Any]
) -> tuple[bool, str]:
    if candidate["recall"] < p["min_recall"]:
        return False, f"recall {candidate['recall']:.3f} below gate {p['min_recall']}"
    if candidate["precision"] < p["min_precision"]:
        return False, f"precision {candidate['precision']:.3f} below gate {p['min_precision']}"
    if current is None:
        return True, "no production model yet"
    worse = [m for m in COMPARED if candidate[m] < current[m]]
    if worse:
        detail = ", ".join(f"{m} {candidate[m]:.3f} < {current[m]:.3f}" for m in worse)
        return False, f"worse than production: {detail}"
    return True, "no regression against production on " + " and ".join(COMPARED)


def current_production(client: MlflowClient, name: str, alias: str) -> dict[str, Any] | None:
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None
    m = client.get_run(mv.run_id).data.metrics
    return {
        "version": mv.version,
        "run_id": mv.run_id,
        **{k: m[f"test_detectable_{k}"] for k in (*COMPARED, "precision")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    p, tp = params["promote"], params["train"]

    model_path = resolve(tp["model"])
    model = Cascade.load(model_path)
    metrics = json.loads(resolve(params["evaluate"]["metrics"]).read_text())
    candidate = metrics["stage1"]["detectable"]
    run_id = model.metadata["mlflow_run_id"]

    mlflow.set_tracking_uri(tracking_uri(tp))
    client = MlflowClient()
    current = current_production(client, p["registered_model"], p["alias"])
    promoted, reason = decide(candidate, current, p)

    example = pd.read_csv(resolve(params["split"]["output_dir"]) / "val.csv").head(3)
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="cascade",
            python_model=CascadePyfunc(),
            artifacts={"cascade": str(model_path)},
            input_example=example[model.features[:6]],
            registered_model_name=p["registered_model"],
        )
        mlflow.set_tag("promotion", "promoted" if promoted else "rejected")
        mlflow.set_tag("promotion_reason", reason)
    version = str(info.registered_model_version)
    if promoted:
        client.set_registered_model_alias(p["registered_model"], p["alias"], version)
    log.info(
        "%s v%s: %s (%s)",
        p["registered_model"],
        version,
        "PROMOTED" if promoted else "kept out",
        reason,
    )

    model_info = {
        "model_version": f"{p['registered_model']}/v{version}",
        "registered_model": p["registered_model"],
        "registry_version": version,
        "promoted": promoted,
        "mlflow_run_id": run_id,
        "data_version": {k: v for k, v in model.metadata.items() if k.startswith("data_sha256")},
        "git_commit": model.metadata.get("git_commit"),
        "threshold": model.threshold,
        "metrics": {
            "test_detectable": candidate,
            "test_all": metrics["stage1"]["all"],
            "test_stage2_macro_f1": metrics["stage2_macro_f1"],
        },
    }
    resolve(p["model_info"]).write_text(json.dumps(model_info, indent=2) + "\n")
    # Registry versions depend on the local MLflow store, so the tracked report
    # keeps only the decision.
    report = {"promoted": promoted, "reason": reason, "candidate": candidate}
    resolve(p["report"]).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
