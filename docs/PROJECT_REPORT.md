# Project Report

## Title

**AI-Powered Customer Intelligence Platform**

## Abstract

This project develops a reproducible Python workflow and Streamlit dashboard for analyzing the UCI Online Retail transaction data. It covers transaction cleaning, business summaries, RFM customer segmentation, historical churn classification, and customer future-revenue modeling. A two-stage customer-value approach estimates purchase probability and revenue conditional on purchasing. Evaluation includes a random customer-level holdout and a separate time-based experiment. Results are specific to the supplied historical data and are not production forecasts.

## Problem and Objectives

Retail transaction records can be difficult to translate into customer-level insights. The project aims to:

- Produce consistent cleaned transaction and customer-level datasets.
- Summarize revenue, orders, customers, countries, and monthly activity.
- Group customers according to Recency, Frequency, and Monetary behavior.
- Demonstrate a defined historical no-purchase churn target.
- Estimate future revenue using purchase probability and conditional revenue.
- Make the analyses explorable through a local dashboard.

## Data and Preparation

The source is the UCI Online Retail workbook. The cleaning workflow removes records without a customer identifier or description, cancellation invoices, and records with nonpositive quantities or prices. Revenue is calculated per transaction line as quantity multiplied by unit price.

Customer prediction datasets use December 1, 2010 through June 30, 2011 as the observation window and July 1 through September 30, 2011 as the target window. Churn is defined as no purchase in the target window. Future customer value is the total retained revenue in that window. The fixed historical periods make the work reproducible but limit its applicability to current customer behavior.

## Implementation

The system is implemented in Python with pandas and scikit-learn. The code is organized into data preparation, analytics, and model scripts, with the Streamlit application in `app/main.py`.

- **Business overview:** revenue, customer and invoice counts, average order value, monthly revenue, country revenue, and recent transactions.
- **Segmentation:** log-transformed and standardized RFM features followed by four-cluster K-Means.
- **Churn:** logistic regression and random forest pipelines trained from customer-level features; the dashboard displays probabilities from the logistic-regression pipeline.
- **Customer value:** a random-forest classifier estimates purchase probability, and a random-forest regressor estimates log revenue for purchasers. Their outputs are combined into expected revenue.
- **Evaluation:** model scripts report classification and regression metrics; temporal value validation compares the two-stage estimate to a training-mean baseline.

## Results

The saved random-split two-stage evaluation contains 592 test customers, with MAE of £348.35 and RMSE of £733.64. Actual average revenue was £476.37 while predicted average revenue was £362.46, indicating underprediction of the test-set mean in that run.

The saved temporal evaluation contains 2,960 test customers. The two-stage model achieved MAE of £477.00 and RMSE of £2,299.43, compared with £801.86 MAE and £3,224.96 RMSE for the training-mean baseline. Actual average revenue was £622.00, and the model predicted an average of £594.08. Purchase ROC-AUC was 0.7500. These figures show that the two-stage model outperformed that baseline on the recorded temporal split, not that it will do so on unseen future data generally.

A separate single-stage comparison artifact reports a random-forest MAE of £356.43 and RMSE of £805.33, versus a training-mean baseline MAE of £670.58 and RMSE of £1,108.83. Ridge regression performed much worse in that stored comparison. These results belong to a different experiment and should not be interpreted as metrics for the dashboard's two-stage model.

The project does not currently persist a churn metric summary as a CSV, so no numeric churn performance claim is made here. The churn training script prints the relevant classification metrics when run.

## Limitations

- The source data and prediction windows are historical; they do not reflect present-day retail behavior.
- The churn outcome is an operational definition based on a three-month period, not proof of permanent customer loss.
- Random customer-level splits can mix customers from the same calendar period between training and testing. The temporal validation is more relevant to time-forward use but covers a single split.
- The dashboard applies models to prepared historical customer datasets that contain known target values. It is for exploration and demonstration, not live scoring.
- Cluster IDs have no inherent ranking or business meaning without reviewing cluster profiles.
- There is no automated test suite in the current project, and saved model/data artifacts can become stale if upstream code or data changes.

## Conclusion

The project demonstrates an end-to-end customer analytics workflow, from transaction cleaning through segmentation and predictive analysis to an interactive dashboard. The saved temporal customer-value result is better than a simple training-mean baseline for its chosen historical window, while the random-split result shows the model can underpredict average value. Further work should prioritize repeatable automated tests, multiple time-based validation windows, model calibration and error analysis, and a scoring workflow that does not rely on future labels being present.

## Reproduction

From the repository root, install `requirements.txt`, run the cleaning and relevant preparation/training scripts described in `README.md`, then launch the dashboard with:

```bash
python -m streamlit run app/main.py
```

Evaluation artifacts referenced in this report are under `data/processed/`.
