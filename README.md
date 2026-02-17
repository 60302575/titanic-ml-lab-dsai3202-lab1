# DSAI3202 -- Lab 3: Data Preprocessing in Azure

## Overview

This lab implements a data preprocessing pipeline using Azure Databricks
and Azure Data Lake Storage Gen2 following the Medallion Architecture
(Bronze → Silver → Gold).

## Pipeline Steps

1.  Load and clean review data (remove nulls, validate ratings, clean
    text).
2.  Enrich reviews with product metadata (asin, title, brand, price).
3.  Create a curated Gold dataset for analytics and ML.

## Technologies Used

-   Azure Databricks\
-   Apache Spark (PySpark)\
-   Azure Data Lake Storage Gen2\
-   Parquet\
-   Databricks Jobs

## Result

An end-to-end ETL pipeline that produces an analytics-ready Gold dataset
(features_v1).
