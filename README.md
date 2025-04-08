# Loan Default Prediction

A machine learning pipeline to predict the likelihood of a customer defaulting on a loan, developed as part of the **MIT Applied Data Science Program Capstone**.

---

<details>
<summary><strong>Problem Statement</strong></summary>

**Why it matters:**
Financial institutions need to assess borrower risk accurately. Approving loans for high-risk applicants leads to financial loss; rejecting low-risk ones limits growth.

**Objective:**
Develop a predictive model that balances **recall** (catching defaulters) and **precision** (avoiding false alarms) to guide smarter lending.

</details>

---

<details>
<summary><strong>Project Pipeline</strong></summary>

![alt text](image.png)
</details>

---

<details>
<summary><strong>Dataset Overview</strong></summary>

- **Target Variable:** `BAD` — 1 = Default, 0 = No Default  
- **Key Features:**  
  `LOAN`, `MORTDUE`, `VALUE`, `YOJ`, `DEROG`, `DELINQ`, `CLAGE`, `NINQ`, `CLNO`, `DEBTINC`, `JOB`, `REASON`

</details>

---

<details>
<summary><strong>Key Insights from EDA</strong></summary>

### **Loan Amount (LOAN):**
- Defaulters tend to have slightly lower loan amounts
- High number of outliers → Outlier treatment required

### **Mortgage Due (MORTDUE) and Property Value (VALUE):**
- Right-skewed distributions
- Defaulters tend to have lower collateral value → potential risk indicator

### **Years on Job (YOJ):**
- Lower tenure for defaulters → employment stability is relevant

### **DEROG & DELINQ (Credit History):**
- Higher frequency among defaulters → highly predictive

### **CLAGE (Credit Line Age):**
- Non-defaulters have longer credit history

### **NINQ (Recent Inquiries):**
- Defaulters have more recent credit checks → financial instability

### **CLNO (Credit Lines):**
- Similar distributions, with more variation among defaulters

### **DEBTINC (Debt-to-Income Ratio):**
- Higher ratios are typical for defaulters

**Preprocessing Actions Taken:**
- Median imputation (YOJ, grouped by JOB)
- Capping outliers (IQR method)
- Encoding categorical features
- Skewness treatment

</details>

---

<details>
<summary><strong>Model Comparison (Before and After Tuning)</strong></summary>

| Model                | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 0.64     | 0.31      | 0.65   | 0.42     | 0.69 |
| Decision Tree       | 0.84     | 0.60      | 0.54   | 0.57     | 0.72 |
| Tuned Decision Tree | 0.79     | 0.49      | 0.84   | 0.62     | 0.87 |
| Random Forest       | 0.89     | 0.80      | 0.62   | 0.70     | 0.94 |
| Tuned Random Forest | 0.79     | 0.49      | 0.83   | 0.61     | 0.88 |

**Notes:**
- **Tuning significantly improved recall** for both DT and RF.
- Random Forest had strong generalization on test set.
- Logistic Regression performed poorly due to linear assumptions.

</details>

---

<details>
<summary><strong>Final Recommendation</strong></summary>

**Best Model:** `Tuned Random Forest`

- Chosen due to **high recall (0.83)** and **balanced AUC (0.88)**
- Well-suited for credit risk settings where identifying defaulters is critical
- Good trade-off between overfitting and generalization

</details>

---

<details>
<summary><strong>Tools & Libraries</strong></summary>

- **Language:** Python 3.12  
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Viz: `matplotlib`, `seaborn`, `missingno`, `plotly`  
  - ML: `scikit-learn`
  - Utilities: `tabulate`, `nbconvert`

</details>

---

<details>
<summary><strong>Project Structure</strong></summary>

```bash
LoanDefaultPrediction/
├── LoanDefaultPrediction.ipynb     # Main notebook
├── LoanDefaultPrediction.html      # Exported version
├── README.md                       # Project summary
├── requiments.txt                  # Necessary libraries
├── .gitignore                      # Not push unwanted files
└── hmeq.csv                        # Input data files
</details>
