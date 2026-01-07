import json
from pathlib import Path
import argparse


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="monitoring/baseline_metrics.json")
    parser.add_argument("--latest", default="monitoring/latest_prod_metrics.json")
    parser.add_argument("--drift", default="monitoring/latest_drift_summary.json")
    parser.add_argument("--churn_rate_delta", type=float, default=0.15)  # 15% absolute delta
    parser.add_argument("--drift_share_threshold", type=float, default=0.30)  # 30% drifted cols
    parser.add_argument("--new_data_dir", default="data/new")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    latest_path = Path(args.latest)
    drift_path = Path(args.drift)
    new_data_dir = Path(args.new_data_dir)

    reasons = []

    # 0) Manual override
    if args.force:
        reasons.append("FORCE_RETRAIN=true")
        print(json.dumps({"retrain": True, "reasons": reasons}, indent=2))
        return 0

    # 1) New data available (simple & effective)
    if new_data_dir.exists():
        files = [p for p in new_data_dir.glob("*") if p.is_file()]
        if len(files) > 0:
            reasons.append(f"new_data_detected: {len(files)} file(s) in {new_data_dir}")
            print(json.dumps({"retrain": True, "reasons": reasons}, indent=2))
            return 0

    # 2) Drift trigger (if you generate a summary)
    if drift_path.exists():
        drift = load_json(drift_path)
        share = float(drift.get("share_drifted_columns", 0.0))
        if share >= args.drift_share_threshold:
            reasons.append(f"data_drift: share_drifted_columns={share:.2f} >= {args.drift_share_threshold:.2f}")
            print(json.dumps({"retrain": True, "reasons": reasons}, indent=2))
            return 0

    # 3) Metric trigger (predicted churn rate vs baseline)
    if baseline_path.exists() and latest_path.exists():
        baseline = load_json(baseline_path)
        latest = load_json(latest_path)

        b = baseline.get("predicted_churn_rate")
        l = latest.get("predicted_churn_rate")

        if b is not None and l is not None:
            b = float(b)
            l = float(l)
            delta = abs(l - b)
            if delta >= args.churn_rate_delta:
                reasons.append(f"metric_shift: |predicted_churn_rate - baseline| = {delta:.2f} >= {args.churn_rate_delta:.2f}")
                print(json.dumps({"retrain": True, "reasons": reasons}, indent=2))
                return 0

    # Default: do not retrain
    reasons.append("no_trigger_matched")
    print(json.dumps({"retrain": False, "reasons": reasons}, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
