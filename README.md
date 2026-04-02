# Assignment 2 — Model Training & Automation with Azure ML

## Overview
This assignment builds a full end-to-end MLOps pipeline for Amazon Electronics review sentiment classification:
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

**Best config:** C=9.007, max_iter=500, solver=lbfgs

## Results

| Split      | Accuracy | F1     | Precision | Recall |
|------------|----------|--------|-----------|--------|
| Deployment | 0.8669   | 0.9212 | 0.8962    | 0.9476 |

54,025 deployment reviews were sent to the live endpoint in batches and evaluated against true labels.

## Feature Experiments
| Run | Features          | Notes |
|-----|-------------------|-------|
| 1   | SBERT only        | Dense embeddings |
| 2   | SBERT + TF-IDF    | Dense + sparse |
| 3   | All features      | Best performance |

## Bonus Question
The thing done **not correctly** is that the **test set is passed into every sweep trial**. In correct MLOps, the test set must be completely held out and only used once — after the best model is chosen using the validation set. Running the test set through every hyperparameter trial leaks information and makes the final test accuracy an overly optimistic estimate.

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
│   ├── train_job.yml         # Training job definition
│   ├── sweep_job.yml         # Hyperparameter sweep job
│   ├── deployment.yml        # Endpoint deployment config
│   └── data_asset_*.yml      # Data asset registrations
├── components/               # Fixed Lab 4 components
└── pipelines/
    └── feature_pipeline.yml  # Updated pipeline with all 4 splits
```
