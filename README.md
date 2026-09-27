# ML Training and Inference Pipeline

A reproducible tabular classification example using a scikit-learn `Pipeline`, stratified holdout, preprocessing fitted on training data only, model persistence, and an optional FastAPI prediction endpoint. The generated dataset is synthetic; reported scores from it are **not** customer churn results.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m ml_pipeline.train --output model.joblib
python -m unittest discover -s tests -v
MODEL_PATH=model.joblib uvicorn ml_pipeline.api:app --reload
```

For a prediction, POST `{"tenure_months":12,"monthly_spend":55,"support_tickets":2}` to `/predict`. Training writes a joblib artifact and a JSON metrics file next to it. `MODEL_PATH` must point to a trusted artifact; joblib files should never be loaded from untrusted sources.

## Design

Synthetic features are tenure, monthly spend, and support tickets. A fixed seed permits reproducible demonstration. The model uses median imputation, standard scaling, and logistic regression. Evaluation includes accuracy, precision, recall, ROC AUC, and class balance on held-out samples. No performance claim is made about real customers. For an actual deployment, add a representative governed dataset, temporal validation, drift monitoring, calibration, and access control.
