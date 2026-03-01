# Lab 4 - Text Feature Engineering with Azure ML

## Overview
This lab transforms raw Amazon Electronics review text into machine learning features using Azure ML Pipelines and registers them in the Azure ML Feature Store.

## Sampling Strategy
To avoid temporal drift, reviews were sampled using stratified sampling by year. This ensures all time periods are equally represented.

## Components
- split_dataset: Splits data into train/val/test (70/15/15)
- normalize_text: Lowercases, removes URLs, numbers, punctuation
- length_features: Computes word and character counts
- sentiment_features: Extracts VADER sentiment scores
- tfidf_features: Creates TF-IDF features (fit on train only to avoid leakage)
- sbert_embeddings: Creates semantic embeddings using all-MiniLM-L6-v2
- merge_features: Merges all features on asin and reviewerID

## Feature Store
Features registered as amazon_review_text_features linked to AmazonReview entity.
