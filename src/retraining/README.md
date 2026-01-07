# Retraining Strategy (Lifecycle)

This project is designed to support the full ML lifecycle: training → deployment → monitoring → retraining.

## When to retrain?
Retraining is triggered when at least one condition is met:

1) **New labeled data is available**
- A new CSV dataset is provided with the target column (Churn).

2) **Performance degradation**
- Example rule: churn recall < 0.70 on a recent evaluation dataset.

3) **Data drift detection** (future work)
- If feature distributions significantly shift compared to training data.

## Retraining steps
1) Load the latest labeled dataset
2) Split train/test
3) Train a new model using the same preprocessing pipeline approach
4) Evaluate with business-driven metrics (ROC-AUC + churn recall)
5) Log metrics, artifacts and the model to MLflow
6) Promote the new model if it improves the selected KPI (churn recall)

## Model promotion rule (simple)
- Promote if: `recall_churn_new >= recall_churn_current` AND `roc_auc_new >= 0.82`
- Otherwise: keep the current production model

## Notes
- Automated orchestration (Airflow/Kubeflow) is not implemented here (future work),
  but the retraining script is structured to be schedulable as a job.
