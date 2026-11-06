# Heart Disease Prediction — Streamlit

This package includes the actual trained model uploaded by the user, not a newly trained substitute.

## Run locally

Use Python 3.11 or 3.12 and create a virtual environment.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `heart_disease_logistic_regression.pkl`, and `requirements.txt` to its root.
2. On Streamlit Community Cloud, create a new app from that repository and select `app.py`.
3. Select Python 3.12 if the hosting settings allow a Python-version choice.
4. Deploy and test several sample inputs.

The model was serialized with scikit-learn 1.6.1, so `requirements.txt` pins that version.

## Important limitations

This is an educational/research demonstration, not a clinical diagnostic tool. Model probabilities are not externally calibrated, and model generalization varied by hospital. Avoid entering identifiable patient information.

## Verification

The Python source was syntax-checked. Full end-to-end inference testing was not possible in the packaging environment because it has a different scikit-learn version and cannot download the compatible one. Test the deployed app before sharing it.
