# Assignment 2 — Model Training & Automation with Azure ML

## Pipeline
```
code push → Azure DevOps CI → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

## Dataset Splits
| Split      | Size | Purpose                                         |
|------------|------|-------------------------------------------------|
| Train      | 60%  | Model training                                  |
| Validation | 15%  | Hyperparameter tuning                           |
| Test       | 15%  | Final offline evaluation                        |
| Deployment | 10%  | Most recent reviews by review_year (data drift) |

## Model
Logistic Regression with saga solver.

## Features
| Group     | Columns                                           |
|-----------|---------------------------------------------------|
| SBERT     | sbert_0 to sbert_383 (384 dims)                  |
| TF-IDF    | tfidf_* (100 features, fit on train only)         |
| Sentiment | sentiment_pos, neg, neu, compound                 |
| Length    | review_length_words, review_length_chars          |

## Results
| Split      | Accuracy | AUC | F1  | Precision | Recall |
|------------|----------|-----|-----|-----------|--------|
| Train      | ???      | ??? | ??? | ???       | ???    |
| Validation | ???      | ??? | ??? | ???       | ???    |
| Test       | ???      | ??? | ??? | ???       | ???    |
| Deployment | ???      | -   | ??? | ???       | ???    |

## Feature Experiments
| Run | Features       | Val Accuracy |
|-----|----------------|--------------|
| 1   | SBERT only     | ???          |
| 2   | SBERT + TF-IDF | ???          |
| 3   | All features   | ???          |

## Bonus Question Answer
The thing done not correctly is that the test set is passed into every sweep trial. In a correct MLOps workflow the test set must remain completely held out and only evaluated once after the best model is selected using validation metrics. Passing test data through every hyperparameter trial causes data leakage making the final reported test accuracy an optimistic and unreliable estimate of true generalization. Only val_accuracy should drive the sweep objective.

## Commands
```bash
# Re-register updated components
az ml component create -f components/split_dataset/component.yml
az ml component create -f components/tfidf_features/component.yml
az ml component create -f components/length_features/component.yml
az ml component create -f components/merge_features/component.yml

# Rerun feature pipeline
az ml job create --file pipelines/feature_pipeline.yml

# Register data assets
az ml data create -f jobs/data_asset_train.yml
az ml data create -f jobs/data_asset_val.yml
az ml data create -f jobs/data_asset_test.yml
az ml data create -f jobs/data_asset_deploy.yml

# Manual training job
az ml job create --file jobs/train_job.yml

# Sweep job
az ml job create --file jobs/sweep_job.yml

# Register model
az ml model create \
  --name amazon-review-sentiment-model \
  --path azureml://jobs/<JOB_NAME>/outputs/model_output \
  --type custom_model

# Create endpoint and deploy
az ml online-endpoint create --name amazon-review-endpoint --auth-mode key
az ml online-deployment create --file jobs/deployment.yml --all-traffic

# Invoke endpoint
python src/invoke_endpoint.py \
  --deploy_data <path/to/deploy/data.parquet> \
  --endpoint_url <url> \
  --api_key <key>

# DELETE ENDPOINT WHEN DONE
az ml online-endpoint delete --name amazon-review-endpoint --yes
```
