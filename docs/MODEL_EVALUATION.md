# Model Evaluation

## Summary

The repository contains several evaluation artifacts from historical experiments. The figures below are read from the generated CSV files in `data/processed/`; they describe those runs and should not be treated as independently reproduced or current performance guarantees.

## Two-Stage Customer Value Model: Random Split

Source: `data/processed/two_stage_customer_value_results.csv`.

| Test customers | Actual average revenue | Predicted average revenue |     MAE |    RMSE |
| -------------: | ---------------------: | ------------------------: | ------: | ------: |
|            592 |                £476.37 |                   £362.46 | £348.35 | £733.64 |

The training script uses a random customer-level 80/20 split. The predictions understate the average revenue in this test split. A random split does not establish performance on a later calendar period.

## Two-Stage Customer Value Model: Temporal Split

Source: `data/processed/customer_value_temporal_validation.csv`.

| Method                 | Test customers | Actual average revenue | Predicted average revenue |     MAE |      RMSE | Purchase ROC-AUC |
| ---------------------- | -------------: | ---------------------: | ------------------------: | ------: | --------: | ---------------: |
| Training-mean baseline |          2,960 |                £622.00 |                   £634.46 | £801.86 | £3,224.96 |   Not applicable |
| Two-stage model        |          2,960 |                £622.00 |                   £594.08 | £477.00 | £2,299.43 |           0.7500 |

This saved result shows lower MAE and RMSE than the training-mean baseline for the specified July-September 2011 target period. Purchase ROC-AUC measures the classifier's ranking of purchasers versus non-purchasers; it does not measure the accuracy of revenue amounts. The evaluation is one historical split on this dataset and may not generalize to other periods or customers.

## Separate Single-Stage Comparison

Source: `data/processed/customer_value_model_results.csv`.

| Model                  |        MAE |        RMSE |
| ---------------------- | ---------: | ----------: |
| Ridge Regression       | £24,309.59 | £580,934.76 |
| Random Forest          |    £356.43 |     £805.33 |
| Training-Mean Baseline |    £670.58 |   £1,108.83 |

This is a separate single-stage experiment, not the two-stage model shown in the dashboard's customer-value page. In this artifact the Ridge errors are substantially larger than those of the other listed models. Treat the saved figures as outputs of the recorded experiment; rerun the relevant training and evaluation workflow to reproduce them from the current data and environment.

## Churn Model

`src/models/train_churn_model.py` prints accuracy, ROC-AUC, a classification report, and a confusion matrix for a stratified random customer-level holdout. The current repository does not contain a dedicated persisted churn evaluation-results CSV, so this document does not report specific churn scores. Run the training script to produce metrics for the current data and installed dependencies.

## Interpretation and Reproducibility

- MAE is the average absolute prediction error in revenue units; RMSE penalizes larger errors more heavily.
- The temporal validation is more informative about later-period use than a random customer split, but it uses only one fixed historical split.
- The dashboard includes historical actual future revenue in its value results because the packaged dataset already contains target labels. It is a demonstration and analysis view, not a prospective scoring workflow.
- To reproduce results, regenerate the required customer dataset, run the corresponding training/evaluation script from the repository root, and compare the newly written CSV. Differences can follow from modified input data, code, or dependency versions.
