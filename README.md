# AI-Powered Customer Intelligence Platform

## 🚀 Live Demo

Try the deployed application here:

[AI Customer Intelligence Platform](https://ai-customer-intelligence-codexplorer.streamlit.app/)

A Python analytics project and Streamlit dashboard for exploring the UCI Online Retail transaction dataset. It prepares customer-level data, summarizes business performance, segments customers with RFM features, estimates historical churn risk, and demonstrates future customer value modeling.

> **Scope:** This is an educational, historical analysis. The churn label and revenue target are defined using 2011 transaction windows, and the dashboard applies models to prepared historical data. Outputs are not current customer forecasts or evidence of production model performance.

## What It Does

- **Business overview:** reports retained revenue, distinct customers and orders, average order value, monthly revenue, and the highest-revenue countries.
- **RFM analysis and segmentation:** calculates Recency, Frequency, and Monetary metrics, then uses scaled, log-transformed RFM values and K-Means to group customers into four clusters.
- **Churn analysis:** defines a customer as churned when they make no purchase during the three-month prediction period. Trains logistic regression and random forest classifiers and displays logistic-regression risk estimates.
- **Customer value analysis:** trains a two-stage model: a purchase classifier followed by a revenue regressor for purchasers. Expected revenue is calculated as purchase probability multiplied by estimated revenue if purchasing.
- **Exploration and evaluation scripts:** include data checks, cluster comparisons, model evaluation, prediction analysis, and a separate temporal customer-value validation experiment.

## Project Layout

```text
app/
	main.py                         Streamlit dashboard
data/
	raw/Online Retail.xlsx          Source workbook expected by the pipeline
	processed/                      Prepared datasets, reports, and saved models
src/
	data/                           Cleaning, exploration, and data checks
	analytics/                      Business metrics and RFM calculations
	models/                         Segmentation, churn, and value workflows
docs/                             Reserved for project documentation
notebooks/                        Reserved for notebooks
powerbi/                          Reserved for Power BI materials
sql/                              Reserved for SQL materials
tests/                             No automated tests are currently present
requirements.txt                  Pinned Python package dependencies
```

The `docs`, `notebooks`, `powerbi`, `sql`, and `tests` folders are currently empty. The files in `data/processed` are generated outputs; the scripts below describe how to reproduce the main dashboard inputs and models.

## Requirements

- Python with versions compatible with the pinned packages in `requirements.txt`. The current development environment uses Python 3.14.2; the project does not declare a broader supported-version range.
- The source workbook at `data/raw/Online Retail.xlsx`. The cleaning script expects the UCI Online Retail columns, including `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

The dataset is the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). Place its workbook at the path above if it is not already available in your checkout.

## Setup

Run commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Then install dependencies with `python -m pip install -r requirements.txt`.

## Rebuild the Dashboard Inputs

Run the scripts from the repository root. Data preparation and model training are separate steps, so create each dataset before running scripts that consume it.

### 1. Clean the source data

```bash
python src/data/explore_data.py       # Optional: inspect the source workbook
python src/data/clean_data.py
python src/data/verify_data.py        # Optional: inspect the cleaned CSV
```

The cleaning step removes rows without a customer ID or description, cancelled invoices, and rows with nonpositive quantity or unit price. It calculates line revenue as `Quantity * UnitPrice` and writes `data/processed/cleaned_retail_data.csv`.

### 2. Prepare customer segments

```bash
python src/analytics/rfm_analysis.py
python src/models/customer_segmentation.py
```

These scripts create `customer_rfm.csv`, then `customer_segments.csv`, `kmeans_model.joblib`, and `rfm_scaler.joblib` in `data/processed/`.

### 3. Prepare and train the churn model

```bash
python src/models/prepare_churn_data.py
python src/data/check_churn_data.py   # Optional: inspect churn labels and features
python src/models/train_churn_model.py
```

The observation window is December 1, 2010 through June 30, 2011; the prediction window is July 1 through September 30, 2011. The label is `1` when a customer made no purchase in the prediction window and `0` otherwise. The training script saves logistic-regression and random-forest pipelines in `data/processed/`.

### 4. Prepare and train the customer value models

```bash
python src/models/prepare_customer_value_data.py
python src/models/train_two_stage_customer_value_model.py
```

This creates `customer_value_data.csv`, the purchase classifier, the conditional revenue model, and `two_stage_customer_value_results.csv` in `data/processed/`. The training script evaluates on a random customer-level holdout split. A separate temporal evaluation is available with:

```bash
python src/models/evaluate_customer_value_temporal.py
```

### 5. Launch the dashboard

```bash
python -m streamlit run app/main.py
```

Open the local URL printed by Streamlit. The app expects the cleaned transaction CSV plus the customer segments, churn dataset and logistic churn model, value dataset, and both two-stage value models. Generate the corresponding artifacts above before opening a page whose inputs are missing.

## Other Analysis Scripts

Run these from the repository root after generating their required data and models:

| Script                                             | Purpose                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `src/analytics/customer_analytics.py`              | Print business-level revenue, customer, order, country, and monthly summaries. |
| `src/models/compare_segmentation.py`               | Compare three- and four-cluster RFM solutions.                                 |
| `src/models/evaluate_segmentation.py`              | Compare K-Means cluster counts using silhouette score and inertia.             |
| `src/models/analyze_churn_model.py`                | Analyze the saved churn model and its evaluation.                              |
| `src/models/train_customer_value_model.py`         | Train the separate single-stage customer-value model.                          |
| `src/models/compare_customer_value_models.py`      | Compare revenue-model approaches.                                              |
| `src/models/evaluate_customer_value_model.py`      | Evaluate the single-stage model on its holdout split.                          |
| `src/models/analyze_customer_value_predictions.py` | Inspect single-stage customer-value prediction errors and ranges.              |
| `src/models/inspect_customer_value_data.py`        | Summarize prepared value features and targets.                                 |
| `src/models/predict_customer_value.py`             | Print a two-stage value estimate for one prepared customer.                    |
| `src/models/generate_customer_value_report.py`     | Generate a two-stage prediction report CSV.                                    |

The single-stage and two-stage customer-value scripts are separate experiments and use different saved models. The dashboard uses the two-stage models.

## Generated Files

The main scripts write artifacts to `data/processed/`, including:

- `cleaned_retail_data.csv`, `customer_rfm.csv`, and `customer_segments.csv`
- `customer_churn_data.csv`, `logistic_churn_model.joblib`, and `random_forest_churn_model.joblib`
- `customer_value_data.csv`, `customer_purchase_classifier.joblib`, and `customer_positive_revenue_model.joblib`
- Evaluation and reporting CSVs, plus the K-Means model and RFM scaler

The exact files depend on which optional analysis scripts have been run. Re-run the relevant preparation script when the source data changes, then retrain downstream models so their features match the regenerated data.

## Interpretation and Limitations

- The churn target means **no recorded purchase in one historical three-month window**; it does not establish permanent churn.
- The standard churn and customer-value training scripts use random customer-level holdout splits. These are preliminary evaluations and do not by themselves demonstrate performance on later calendar periods. The temporal customer-value script provides a separate time-based experiment.
- Customer cluster numbers are identifiers, not ordered customer rankings. Interpret clusters using their RFM profiles.
- The dashboard's customer-value page displays actual future revenue alongside model estimates because the bundled dataset contains historical target labels. This makes it useful for demonstration and analysis, not prospective scoring.
- Model files created with joblib are Python/scikit-learn artifacts. Use trusted files produced by this project; loading joblib files from untrusted sources can execute code.
- There is no automated test suite in the current project. Data verification and model evaluation are provided as standalone scripts.
