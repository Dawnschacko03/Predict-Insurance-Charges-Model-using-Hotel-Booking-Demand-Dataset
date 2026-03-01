# Hotel Booking Cancellation Prediction

## End-to-End Machine Learning Pipeline for Revenue Risk Mitigation

I built a cancellation prediction model using classification algorithms. Logistic Regression served as a baseline. Random Forest improved performance by capturing non-linear patterns. XGBoost achieved the best ROC-AUC due to its gradient boosting approach and regularization capability.


📌 Executive Summary

This project builds a production-ready machine learning system to predict hotel booking cancellations using historical reservation data. The objective is to proactively identify high-risk bookings and enable data-driven strategies such as:

Dynamic pricing

Deposit optimization

Customer retention campaigns

Revenue protection planning

The final solution compares multiple models and selects the best-performing algorithm based on business-relevant metrics.


🧠 Problem Framing

Booking cancellations directly impact revenue forecasting and occupancy planning.
The task was formulated as a binary classification problem:

Predict whether a booking will be canceled (is_canceled = 1) or not (0).

Dataset size: 119,210 bookings

Feature space: 52 engineered features

⚙️ Solution Architecture

1️⃣ Data Processing

Handled missing values using median imputation

Validated class balance

Applied feature scaling for linear models

Ensured stratified train-test split

2️⃣ Model Development

Three models were implemented and benchmarked:

Logistic Regression (Baseline)

Random Forest Classifier

XGBoost Classifier

3️⃣ Evaluation Metrics

To ensure business relevance, models were evaluated using:

Accuracy

ROC-AUC

Precision

Recall (critical for detecting cancellations)

F1-Score


##📊 Model Performance

Model	                 Accuracy	        ROC-AUC	      Recall (Canceled=1)

Logistic Regression	    80.79%	         0.85	         0.59

Random Forest	          86.49%	         0.93	         0.76

XGBoost	                83.69%	         0.90	         0.68


🏆 Final Model Selection

Random Forest was selected as the production-ready model due to:

Highest overall accuracy

Strongest ROC-AUC score (0.93)

Superior recall (76%) for cancellation detection

Balanced precision-recall tradeoff

From a business perspective, minimizing missed cancellations was prioritized over marginal accuracy gains.


📈 Key Insights

Long lead times significantly increase cancellation probability.

Customer segment and booking type are strong predictors.

Ensemble models outperform linear models due to non-linear feature interactions.

Recall optimization materially improves risk detection.


📊 Visual Analytics

ROC Curve Comparison

Model Accuracy Benchmark

Confusion Matrix Analysis

Feature Importance (XGBoost)

##🛠 Tech Stack

Python

Pandas / NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn
