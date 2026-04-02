# Assignment 2 — Model Training & Automation with Azure ML

## Overview
Full end-to-end MLOps pipeline for Amazon Electronics review sentiment classification:
```
code push → Azure DevOps CI → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

## Dataset Splits
| Split      | Size | Purpose |
|------------|------|---------|
| Train      | 60%  | Model training |
| Validation | 15%  | Hyperparameter tuning |
| Test       | 15%  | Final offline evaluation |
| Deployment | 10%  | Most recent reviews by review_year (simulates data drift) |

## Model Choice
**Logistic Regression** with `lbfgs` solver.
- Fast on high-dimensional text features (SBERT + TF-IDF)
- Strong baseline for sentiment classification
- Easy to tune and debug through CI pipelines

## Features Used
| Group     | Columns                                  | Dims |
|-----------|------------------------------------------|------|
| SBERT     | sbert_0 to sbert_383                    | 384  |
| TF-IDF    | tfidf_* (fit on train only)             | 100  |
| Sentiment | sentiment_pos, neg, neu, compound        | 4    |
| Length    | review_length_words, review_length_chars | 2    |
| **Total** |                                          | **490** |

## Hyperparameter Tuning
6 trials with random sampling over:
- `C`: uniform(0.01, 10.0)
- `max_iter`: choice([500, 1000, 2000])
- `solver`: choice([saga, lbfgs])

**Best config:** C=9.007, max_iter=500, solver=lbfgs (val_accuracy=0.8508)

## Final Performance

| Split      | Accuracy | AUC    | F1     | Precision | Recall |
|------------|----------|--------|--------|-----------|--------|
| Train      | 0.8014   | 0.4205 | 0.8898 | 0.8015    | 0.9999 |
| Validation | 0.8008   | 0.4263 | 0.8894 | 0.8008    | 0.9999 |
| Test       | 0.8024   | 0.4234 | 0.8904 | 0.8024    | 0.9999 |
| Deployment | 0.8669   | -      | 0.9212 | 0.8962    | 0.9476 |

The deployment split uses the most recent reviews by `review_year`, simulating real production data drift.
The higher deployment accuracy compared to test suggests the model generalizes well to recent reviews.

## Feature Experiments
| Run | Features          | Notes |
|-----|-------------------|-------|
| 1   | SBERT only        | Dense semantic embeddings |
| 2   | SBERT + TF-IDF    | Dense + sparse text features |
| 3   | All features      | Best performance |

## Bonus Question
The thing done **not correctly** is that the **test set is passed into every sweep trial**. In correct MLOps, the test set must be completely held out and only used once after the best model is chosen using the validation set. Running the test set through every hyperparameter trial leaks information and makes the final test accuracy an overly optimistic estimate of true generalization performance. Only `val_accuracy` should drive the sweep objective.

## Lab 4 Fixes Applied
- Added **deployment split** (10% most recent reviews by `review_year`)
- Fixed pipeline to run **all features on all 4 splits** (not just train)
- Fixed `length.py` to **preserve the `overall` column**
- Fixed `merge.py` to **avoid `_x/_y` column conflicts**
- Fixed `sentiment.py` to **download vader_lexicon to writable path**

## Repo Structure
```
├── azure-pipelines.yml       # CI pipeline (triggers on push)
├── src/
│   ├── train.py              # Training script with MLflow logging
│   ├── score.py              # Scoring script for endpoint
│   └── invoke_endpoint.py    # Calls endpoint with deploy dataset
├── env/
│   ├── conda.yml             # Training environment
│   └── inference_conda.yml   # Inference environment
├── jobs/
│   ├── train_job.yml         # Training job (C=9.007, lbfgs, max_iter=500)
│   ├── sweep_job.yml         # Hyperparameter sweep job
│   ├── deployment.yml        # Endpoint deployment config
│   └── data_asset_*.yml      # Data asset registrations
├── components/               # Fixed Lab 4 components
└── pipelines/
    └── feature_pipeline.yml  # Updated: all 4 splits get all features
```

## Commands Reference
```bash
# Register updated components
az ml component create -f components/split_dataset/component.yml
az ml component create -f components/tfidf_features/component.yml
az ml component create -f components/length_features/component.yml
az ml component create -f components/merge_features/component.yml

# Run feature pipeline
az ml job create --file pipelines/feature_pipeline.yml

# Register data assets
az ml data create -f jobs/data_asset_train.yml
az ml data create -f jobs/data_asset_val.yml
az ml data create -f jobs/data_asset_test.yml
az ml data create -f jobs/data_asset_deploy.yml

# Manual training job
az ml job create --file jobs/train_job.yml

# Hyperparameter sweep
az ml job create --file jobs/sweep_job.yml

# Register model
az ml model create \
  --name amazon-review-sentiment-model \
  --path azureml://jobs/<JOB_NAME>/outputs/model_output \
  --type custom_model

# Create endpoint and deploy
az ml online-endpoint create --name amazon-review-60302575 --auth-mode key
az ml online-deployment create --file jobs/deployment.yml --all-traffic

# Delete endpoint when done
az ml online-endpoint delete --name amazon-review-60302575 --yes
```
