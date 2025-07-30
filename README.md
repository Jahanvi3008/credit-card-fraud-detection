# 💳 Credit Card Fraud Detection

Detect fraudulent credit card transactions using machine learning models trained on imbalanced data. This project demonstrates end-to-end classification including data preprocessing, model building, evaluation, and interactive prediction using Streamlit.

## 📂 Project Structure

- `fraud_detection.ipynb` — Jupyter notebook with data exploration, modeling, and evaluation.
- `streamlit_app.py` — Interactive app to simulate real-time fraud predictions.
- `data/` — Placeholder for dataset (link below).
- `images/` — Evaluation plots.

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions with 492 frauds (0.17%)
- **Features**: PCA-transformed + `Amount`, `Time`, `Class` (0 = normal, 1 = fraud)

## 🧠 ML Techniques Used

- Exploratory Data Analysis (EDA)
- Feature Scaling
- Class Imbalance Handling (SMOTE, Undersampling)
- Model Training: Logistic Regression, Random Forest
- Evaluation: Confusion Matrix, AUC, Precision, Recall

## 🚀 Streamlit App

Run the app locally:

```bash
streamlit run streamlit_app.py
```

🌐 Demo: https://streamlit-demo-link-placeholder.com

## 📈 Results

- **Best model**: Random Forest (AUC ≈ 0.98)
- Key metrics optimized for **high recall** (minimize false negatives)

## 📌 Key Learnings

- How to handle imbalanced datasets
- Importance of precision-recall trade-off
- Real-world application in fraud prevention

## 🧑‍💻 Author

- Jahanvi Dave  
- [LinkedIn](https://www.linkedin.com/in/jahanvi-8271a7214) | [GitHub](https://github.com/Jahanvi3008)
