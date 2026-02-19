🫁 Lung Cancer Prediction and Web Application Using Machine Learning

📌 Overview
This project presents a complete end-to-end machine learning pipeline for **lung cancer risk prediction**, integrating data preprocessing, multi-model comparison, performance evaluation, and deployment into an interactive web application.
The system uses structured clinical, demographic, and lifestyle features to predict lung cancer risk and deploys the best-performing model through a Gradio-based web interface for real-time screening.

> ⚠ This system is intended for screening and educational purposes only and does not replace professional medical diagnosis.

🎯 Project Objectives
* Develop a scalable ML pipeline for lung cancer prediction.
* Handle real-world clinical data preprocessing.
* Address class imbalance in medical datasets.
* Compare multiple supervised learning algorithms.
* Perform hyperparameter tuning.
* Analyze feature importance.
* Deploy the best-performing model via a web application.
* Build a complete research-to-deployment workflow.

📊 Dataset Description
* **Total Records:** 460,292 patient records
* **Total Features:** 20
* **Target Variable:** Lung Cancer (Yes / No)
* **Class Imbalance:** ~80% No, ~20% Yes

Feature Categories

| Category          | Examples                                      |
| ----------------- | --------------------------------------------- |
| Demographic       | Age, Gender, Country, Rural/Urban             |
| Behavioral        | Smoking Status, Second-hand Smoke             |
| Environmental     | Air Pollution, Occupational Exposure          |
| Clinical Symptoms | Shortness of Breath, Wheezing, Coughing Blood |
| Medical History   | Family History                                |


🔬 Work Done in This Project

1️⃣ Data Preprocessing
* Separated features and target
* Label encoding of categorical variables
* Safe encoding for unseen categories
* Feature scaling using StandardScaler
* Stratified train-test split
* Class imbalance handling using:
  * Balanced class weights
  * Sample weighting

2️⃣ Machine Learning Models Implemented
We trained and compared the following models:
* Logistic Regression
* Linear Support Vector Machine
* Random Forest Classifier ✅ (Best Performing)
* Gradient Boosting
* LightGBM
* Dummy Classifier (Baseline)

3️⃣ Hyperparameter Optimization
* GridSearchCV
* RandomSearchCV
* Cross-validation
* Weighted scoring metrics

Optimized parameters for Random Forest:
* n_estimators = 200
* max_depth = 15
* min_samples_split = 5
* min_samples_leaf = 2
* class_weight = balanced

4️⃣ Model Evaluation
Models were evaluated using:
* Accuracy
* Precision
* Recall (Sensitivity)
* F1-score
* Confusion Matrix
* ROC Analysis

🏆 Best Model: Random Forest
* Accuracy: ~75%
* Balanced precision-recall performance
* Robust against overfitting
* Strong feature interpretability

📈 Feature Importance Insights
Top predictive features included:
* Smoking Status
* Coughing of Blood
* Shortness of Breath
* Air Pollution Exposure
* Family History

These align well with established clinical knowledge, validating model behavior.

🌐 Web Application Deployment
The trained Random Forest model is deployed using **Gradio**.

Features of Web App:
* Dynamic dropdown inputs for categorical variables
* Numeric fields for continuous features
* Adjusted detection threshold (35%) for higher sensitivity
* Real-time probability output
* Risk classification (Low / Moderate / High)
* Clear medical disclaimer
* Clean UI styling

Screening Strategy: We lowered the prediction threshold to 35% to prioritize sensitivity in a screening context, reducing false negatives.

🧠 System Architecture

Dataset -> Preprocessing -> Model Training -> Model Evaluation -> Feature Importance -> Model Selection -> Web Deployment (Gradio)

🛠 Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Gradio

⚠ Limitations
* Moderate accuracy (~75%)
* Dataset imbalance
* No external validation dataset
* Demo-level deployment (Colab-based)

🔮 Future Improvements
* SMOTE or advanced imbalance handling
* SHAP-based explainability
* Ensemble stacking
* External validation datasets
* Docker-based production deployment
* Secure clinical deployment

📄 Research & Academic Context
This project was developed as part of a Bachelor of Technology Capstone Project.

It demonstrates:
* Full ML lifecycle implementation
* Clinical data handling
* Model benchmarking
* Deployment-ready system design
* Responsible AI communication

🧑‍⚕ Ethical Disclaimer
This system is intended for educational and screening research purposes only. It is not a certified medical device and should not be used for clinical decision-making without professional consultation.

📬 Authors
* Narne Nithin Kumar
* Polisetti Govardhini
* Kalluri Ram Charan Teja Reddy
