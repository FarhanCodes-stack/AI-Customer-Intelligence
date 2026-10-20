# Methodology

## Dataset and Cleaning

The project uses the UCI Online Retail transaction workbook at `data/raw/Online Retail.xlsx`. The cleaning script expects transaction columns including invoice number and date, stock code, description, quantity, unit price, customer ID, and country.

`src/data/clean_data.py` applies these rules:

- Drop records without `CustomerID` or `Description`.
- Exclude cancellation invoices whose invoice number starts with `C`.
- Keep only rows with positive `Quantity` and `UnitPrice`.
- Convert customer IDs to integers.
- Calculate transaction-line revenue as `Quantity * UnitPrice`.

The output is `data/processed/cleaned_retail_data.csv`. This filtering means downstream revenue describes retained positive transaction lines, not gross sales including cancellations or returns.

## Business Metrics

The dashboard and `src/analytics/customer_analytics.py` calculate total retained revenue, distinct customers, distinct invoices, and average order value. Average order value is total retained revenue divided by the number of distinct invoices. Revenue is also grouped by month and country.

## RFM Features and Segmentation

`src/analytics/rfm_analysis.py` aggregates cleaned transactions by customer:

- **Recency:** days between the day after the last transaction in the dataset and that customer's most recent purchase.
- **Frequency:** number of distinct invoices.
- **Monetary:** sum of retained transaction revenue.

`src/models/customer_segmentation.py` uses Recency, Frequency, and Monetary. It clips Frequency to at least 1 and Monetary to at least 0, applies `log1p` to Frequency and Monetary, standardizes the dimensions, and fits K-Means with four clusters (`random_state=42`, `n_init=10`). The scaler and model are saved separately. Cluster IDs are arbitrary labels and must be interpreted through the resulting RFM profiles.

## Churn Dataset and Model

`src/models/prepare_churn_data.py` creates customer-level inputs from two fixed historical periods:

- Observation: December 1, 2010 through June 30, 2011, inclusive.
- Prediction: July 1, 2011 through September 30, 2011, inclusive.

Observation-period features are Recency, Frequency, Monetary, Total Quantity, Unique Products, and Observation Days. First and Last Purchase dates are retained in the prepared CSV for auditing, not selected as model features. The target is `1` if the customer has no purchase in the prediction period and `0` otherwise.

`src/models/train_churn_model.py` makes a stratified random 80/20 customer-level split (`random_state=42`) and trains:

- Logistic regression with median imputation, standard scaling, and a 1,000-iteration limit.
- Random forest with 200 trees and balanced class weights.

The script reports accuracy, ROC-AUC, classification report, and confusion matrix, and saves both models. The dashboard uses the logistic-regression pipeline and labels probabilities of at least 0.5 as higher risk. This label means no observed purchase in the chosen historical window; it is not proof of permanent churn or present-day risk.

## Customer Value Dataset and Two-Stage Model

`src/models/prepare_customer_value_data.py` uses the same fixed observation and prediction periods as the churn preparation. It computes historical features from the observation period only: Recency, Frequency, Monetary, Total Quantity, and Unique Products. It sums revenue per customer in the prediction period as `Future_Revenue`, assigning zero to customers with no future-period transactions.

`src/models/train_two_stage_customer_value_model.py` uses a random 80/20 customer-level split (`random_state=42`):

1. A random-forest classifier predicts whether future revenue is positive.
2. A random-forest regressor is fit only on positive-revenue training examples and predicts `log1p(Future_Revenue)`.
3. Conditional revenue is transformed back with `expm1`; expected revenue is purchase probability multiplied by conditional revenue.

Both forests use 200 estimators and `random_state=42`; the classifier uses `min_samples_leaf=5` and balanced class weights, and the regressor uses `min_samples_leaf=5`. The combined prediction is evaluated with MAE and RMSE and saved with its models.

## Temporal Customer Value Evaluation

`src/models/evaluate_customer_value_temporal.py` performs a separate time-based experiment:

- Training features use December 2010 through March 2011; training targets use April through June 2011.
- Test features use December 2010 through June 2011; test targets use July through September 2011.

It trains the two stages on the earlier period and compares later-period revenue estimates against a training-mean baseline. It reports MAE, RMSE, and purchase ROC-AUC, and writes `customer_value_temporal_validation.csv`. This is a single historical temporal split, not a guarantee of future performance.

## Separate Single-Stage Experiment

The repository also contains a separate single-stage customer-value regressor and comparison/evaluation scripts. Its metrics and model artifacts are distinct from the two-stage workflow used by the dashboard. Results from these experiments should not be conflated.
