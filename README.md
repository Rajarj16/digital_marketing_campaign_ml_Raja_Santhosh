# 📊 Digital Marketing Campaign Engagement Prediction

### Machine Learning & Power BI Integration (Master’s Thesis)

---

## 📌 Project Overview

This repository contains the **machine learning implementation and Power BI dashboard** developed as part of my Master’s Thesis:

**“Analyzing Digital Marketing Campaign Performance Using Data-Driven Analytics and Machine Learning.”**

The project focuses on **predicting social media campaign engagement rates** using historical digital marketing data and identifying the **key factors influencing engagement**.
Machine learning is implemented using **Python (Google Colab / Jupyter Notebook)**, and the resulting predictions are integrated into **Power BI** to create interactive, decision-oriented dashboards.

This work demonstrates an **end-to-end analytics pipeline**, combining predictive modeling with business intelligence.

---

## 🎯 Research Objectives

* Predict **engagement rate** based on historical campaign performance data
* Compare a **baseline model** with a regression-based machine learning model
* Identify **key drivers of engagement** using feature importance analysis
* Integrate ML prediction outputs into **Power BI dashboards**
* Provide **data-driven insights** to support digital marketing decision-making

---

## 🧠 Machine Learning Approach

### Models Implemented

* **Baseline Model:** Dummy Regressor (mean strategy)
* **Predictive Model:** Ridge Regression

### Rationale for Ridge Regression

* Handles multicollinearity in high-dimensional marketing datasets
* Suitable for datasets with mixed numerical and categorical variables
* Provides interpretable coefficients for feature importance analysis

---

## ⚙️ Methodology

### Data Preprocessing

* Missing value handling using **SimpleImputer**
* Feature scaling using **StandardScaler**
* Categorical encoding using **OneHotEncoder**
* Pipeline-based preprocessing using **ColumnTransformer**

### Target Variable

* `engagement_rate`
* Normalized to a **0–1 range** prior to model training

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📊 Machine Learning Outputs

The notebook generates the following outputs:

* **Actual vs Predicted Engagement Rate (Scatter Plot)**
* **Top Feature Importance Bar Chart (Ridge Regression)**
* **Prediction Output File:**

  * `ml_predictions_final.csv`
* **Feature Importance Output File:**

  * `v_ml_features_final.csv`

These outputs form the basis for **Power BI dashboard visualizations**.

---

## 🔗 Integration with Power BI

This project uses a **CSV-based ML-to-Power BI integration approach**, which is common in both academic research and industry analytics workflows.

### Integration Workflow

1. Machine learning models are developed and executed in **Google Colab / Jupyter Notebook**
2. The trained model generates:

   * Predicted engagement rates
   * Feature importance values
3. ML outputs are exported as CSV files:

   * `ml_predictions_final.csv`
   * `v_ml_features_final.csv`
4. CSV files are imported into **Power BI**
5. Interactive dashboards are created to visualize:

   * Actual vs Predicted Engagement Rate
   * Average Predicted Engagement by Platform and Campaign
   * Prediction Error Analysis
   * Key Factors Influencing Engagement

> Machine learning is performed entirely in Python, while Power BI is used strictly as a **visualization and decision-support layer**.

---

## 📊 Power BI Dashboard

The Power BI dashboard demonstrates how machine learning outputs can be operationalized for business insights.

### Dashboard Features

* Campaign performance overview
* Actual vs Predicted Engagement Rate visualization
* Platform-wise and campaign-phase analysis
* Prediction error comparison
* Visual interpretation of ML-driven insights

To reproduce the dashboard results:

1. Run the ML notebook to regenerate CSV output files
2. Open the Power BI `.pbix` file
3. Refresh data sources to load the updated CSV files

---

## 🗂️ Repository Structure

```
├── digital_marketing_campaign_ml_Raja_Santhosh.ipynb   # Machine learning implementation
├── social_media_engagement_data_cleaned.csv            # Cleaned dataset used for ML
├── ml_predictions_final.csv                            # ML prediction outputs
├── v_ml_features_final.csv                             # Feature importance outputs
├── Digital_Marketing_Campaign_Analytics.pbix           # Power BI dashboard
├── README.md                                           # Project documentation
                                          # Project documentation
```

---

## 🛠️ Requirements

* Python 3.8+
* Jupyter Notebook or Google Colab
* Power BI Desktop
* Required Python libraries:

  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  ```

---

## ▶️ How to Run the Project

### Machine Learning

1. Clone the repository:

   ```bash
   git clone <your-github-repository-url>
   cd <repository-folder>
   ```

2. Open the notebook:

   ```bash
   jupyter notebook digital_marketing_campaign_ml_Raja_Santhosh.ipynb
   ```

3. Run all cells **from top to bottom** to:

   * Preprocess data
   * Train baseline and Ridge models
   * Generate ML visuals
   * Export CSV outputs for Power BI

### Power BI

1. Open `Digital_Marketing_Campaign_Analytics.pbix`
2. Refresh data sources
3. Explore ML-enhanced dashboards

---

## 📈 Key Findings

* Engagement rate prediction is challenging due to high variability in social media behavior
* Ridge Regression outperforms the baseline Dummy Regressor
* Engagement is influenced by:

  * Impressions
  * Likes and comments
  * Campaign phase
  * Platform
  * Sentiment-related features
* ML-enhanced dashboards provide **predictive insights beyond descriptive analytics**

---

## 🔒 Ethical Considerations

* Data used is anonymized or synthetic
* No personally identifiable information (PII) is included
* Analysis conducted strictly for academic purposes

---

## 🚀 Future Enhancements

* Experiment with non-linear models (Random Forest, Gradient Boosting)
* Add cross-validation and hyperparameter tuning
* Extend analysis to time-series forecasting
* Deploy ML model as an API for real-time dashboards

---

## 👤 Author

**Raja Ramaraj and Santhosh Selvan**
Master’s Thesis – Digital Marketing Analytics / Data Analytics
