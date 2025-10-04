# Credit Risk Modeling: Can We Predict Loan Defaults Before They Happen?

## Project Overview
This project explores one of the most critical challenges in finance and fintech: loan defaults. Financial institutions face billions in losses every year due to borrowers failing to repay their loans. Striking the right balance between lending responsibly and expanding financial inclusion is an ongoing challenge.  

The guiding question for this project is simple yet powerful:  
*Can I predict the likelihood of a borrower defaulting before the loan is approved?*  

By following the data science lifecycle, I aim to build an end-to-end machine learning pipeline that not only predicts defaults but also provides interpretable insights that credit officers can trust.  

---

## Data Science Lifecycle

### 1. Business Understanding ✅
Loan defaults reduce profitability, increase risk exposure, and limit financial inclusion. The objective of this project is to create a predictive model that identifies high-risk borrowers while minimizing unfair rejection of low-risk applicants. Success will be measured both by model performance (Precision, Recall, F1, ROC-AUC) and by business impact (balanced risk management).

### 2. Analytic Approach ✅
This is framed as a binary classification problem with two outcomes: *default* or *no default*. Because the costs of errors are asymmetric, accuracy alone is not enough. Evaluation will focus on Precision, Recall, F1, and ROC-AUC.  

I will begin with Logistic Regression for interpretability, then extend to ensemble methods such as Random Forests and Gradient Boosting (XGBoost, LightGBM). Interpretability techniques like feature importance and SHAP values will ensure transparency in model predictions.  

### 3. Data Requirements ✅
The model is designed to run at the moment of loan approval, meaning it can only rely on inputs available at application time. This includes borrower attributes (income, employment history, credit score), loan characteristics (amount, term, interest rate, purpose), and contextual factors (application date, channel, region).  

The dataset must be large enough to capture rare but critical default events, clean and reliable to ensure credible insights, and compliant with ethical and regulatory standards. Sensitive attributes will not be used for prediction, but fairness auditing and interpretability will remain central.  

For this project, I will be working with real-world LendingClub loan data, which provides millions of applications and repayment outcomes, making it a suitable foundation for this model.  

### 4. Data Collection & Understanding 

After exploring multiple data sources, I chose the Fannie Mae Single-Family Loan Performance dataset as the foundation for this project.
Because each quarter’s data is several gigabytes in size, I’m starting with Q1 2022 as a complete case study. This allows me to design, clean, and model the entire pipeline using free resources (Kaggle + local environment) before scaling to additional quarters.

The Q1 2022 dataset captures detailed borrower, loan, property, and monthly performance information. Each loan appears across several months, enabling me to engineer a clear target label — default (1) if a loan becomes 90+ days delinquent within 12 months of origination, otherwise no default (0).

Initial findings highlight common data-science challenges: missing values, extreme numeric ranges, and class imbalance between defaulted and non-defaulted loans. These insights will guide feature engineering, imputation, and resampling strategies in the next stage (Data Preparation).
---

## Repository Structure
```

credit\_risk/
│
├── data/                # Raw and processed data (not tracked in GitHub)
│   ├── raw/
│   └── processed/
├── notebooks/           # Jupyter notebooks for exploration and modeling
├── src/                 # Reusable scripts (data prep, training, evaluation)
├── dashboard/           # Streamlit app for deployment
│   └── app.py
├── reports/             # Reports and visualizations
│   └── figures/
├── tests/               # Unit tests for scripts
├── .gitignore
├── requirements.txt
└── README.md

```

---

## Tech Stack
- **Languages**: Python (Pandas, NumPy)  
- **Modeling**: scikit-learn, XGBoost, LightGBM, imbalanced-learn  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Interpretability**: SHAP, Feature Importance  
- **Deployment**: Streamlit  
- **Experimentation**: Jupyter Notebooks  

---

## Project Goals
- Build a predictive model for loan defaults.  
- Extract interpretable insights for business decision-making.  
- Deploy a prototype dashboard for risk assessment.  
- Document the entire process publicly through LinkedIn and Medium posts.  

---

## Progress
- [x] Business Understanding  
- [x] Analytic Approach  
- [x] Data Requirements  
- [ ] Data Collection  
- [ ] Data Understanding  
- [ ] Data Preparation  
- [ ] Modeling  
- [ ] Evaluation  
- [ ] Deployment  

---

## Author
**Francis Blessed Kim**  
- [LinkedIn](https://www.linkedin.com/in/francis-kim-1931681b6/)  
- [GitHub](https://github.com/francisblessedkim)  
- [Medium](https://medium.com/@kimblessedfrancis)  
```

---


