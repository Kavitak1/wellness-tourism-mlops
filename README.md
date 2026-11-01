# Wellness Tourism Package — MLOps pipeline

Predicts whether a Visit with Us customer will purchase the Wellness Tourism Package (`ProdTaken`) before the sales team makes contact.

## Repository layout

```text
wellness_tourism_mlops/
├── data/
│   └── tourism.csv              # registered raw dataset
├── src/
│   ├── validate_data.py         # column contract and data summary
│   ├── prepare_data.py          # cleaning, split, saved train and test sets
│   └── train_model.py           # tuning, experiment log, best model
├── models/                      # best_model.joblib committed by the pipeline
├── reports/                     # experiment log written by training
├── app.py                       # Streamlit scoring app
├── requirements.txt             # deployment dependencies
├── runtime.txt                  # Python version for Streamlit Community Cloud
├── .github/workflows/pipeline.yml
└── Wellness_Tourism_MLOps.ipynb
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.validate_data
python -m src.prepare_data
python -m src.train_model
streamlit run app.py
```

## GitHub Actions

`.github/workflows/pipeline.yml` runs three jobs in order:

1. **Data registration** checks the expected columns and prints a summary.
2. **Data preparation** cleans the CSV, writes the train and test splits, and uploads them as the `processed-data` artifact.
3. **Model building** downloads that artifact, tunes the candidate models, writes the experiment log, and pushes `models/best_model.joblib` back to `main`.

Pushes that only change `models/`, `reports/`, or `data/processed/` do not start another run, so the model commit does not retrain in a loop.

## Streamlit Community Cloud

1. Push this repository to a public GitHub repo.
2. In Streamlit Community Cloud, create an app from that repo.
3. Set the main file to `app.py` and the requirements file to `requirements.txt`.
4. Paste the app URL into the notebook submission section.

## Submission links

- GitHub repository: add the public URL after the first push
- Streamlit app: add the Community Cloud URL after deployment
