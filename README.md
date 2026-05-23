# Income Classification Analysis
### NPTEL Python for Data Science — Certified Course Project
### Author: M. Charulatha | MSc Bioinformatics and Data Science

---

## Objective
Predict whether an individual earns more than 50,000
or less than or equal to 50,000 based on demographic
and employment features using classification models.

---

## Dataset
- Name    : Adult Income Dataset
- Source  : UCI Machine Learning Repository
- Records : 31,978 rows, 13 columns
- Features: age, JobType, education, occupation,
            marital status, gender, capital gain,
            capital loss, hours per week, native country
- Target  : SalStat (less than or equal to 50K or greater than 50K)

---

## Steps Performed
1. Loaded and explored the dataset (EDA)
2. Detected hidden missing values (space-question mark to NaN)
   — 1,816 rows removed — clean dataset: 30,162 rows
3. Analysed relationships using cross-tabulation and correlation
4. Created 9 visualisations to understand feature patterns
5. Encoded target variable and applied One-Hot Encoding
6. Split data: 70% training, 30% testing
7. Built and evaluated 2 classification models
8. Found optimal K for KNN by testing K from 1 to 19

---

## Models and Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 84.27% |
| KNN (k=16) | 84.36% |

---

## Model Evaluation Details

### Logistic Regression
| Metric | Value |
|---|---|
| Accuracy | 84.27% |
| Misclassified Samples | 1,424 |

Confusion Matrix:

|  | Predicted less than or equal to 50K | Predicted greater than 50K |
|---|---|---|
| Actual less than or equal to 50K | 6329 (True Negative) | 494 (False Positive) |
| Actual greater than 50K | 930 (False Negative) | 1296 (True Positive) |

### KNN Classifier (k=16)
| Metric | Value |
|---|---|
| Accuracy | 84.36% |
| Misclassified Samples | 1,415 |

Confusion Matrix:

|  | Predicted less than or equal to 50K | Predicted greater than 50K |
|---|---|---|
| Actual less than or equal to 50K | 6389 (True Negative) | 434 (False Positive) |
| Actual greater than 50K | 981 (False Negative) | 1245 (True Positive) |

### Interpretation
- KNN correctly identified more low earners (6389 vs 6329)
- Logistic Regression identified slightly more high earners (1296 vs 1245)
- Both models perform comparably — KNN edges ahead by 0.09%
- Higher False Negatives in both models are due to class imbalance
  — 75% of the dataset earns less than or equal to 50K

---

## Visualisations

| Plot | Finding |
|---|---|
| Salary Status Distribution | Class imbalance — far more people earn less than or equal to 50K |
| Age Distribution | Working population peaks between age 25 to 50 |
| Age vs Salary Status | Higher earners are older — median age 44 vs 34 |
| JobType vs Salary Status | Private sector dominates all income levels |
| Education vs Salary Status | Higher education strongly predicts income above 50K |
| Occupation vs Salary Status | Exec-managerial and Prof-specialty earn most above 50K |
| Capital Gain Distribution | Highly right-skewed — most people have zero capital gain |
| Capital Loss Distribution | Highly right-skewed — most people have zero capital loss |
| Hours Per Week vs Salary | Higher earners tend to work more hours per week |

---

## Key Insights
- Education is the strongest predictor of income above 50K
- Exec-managerial and Prof-specialty occupations earn the most
- Older individuals are more likely to earn above 50K
- People working more hours per week tend to earn more
- Capital gain is a strong but skewed predictor
- KNN with k=16 performs slightly better than Logistic Regression
- Class imbalance affects model performance on the minority class

---

## Tools and Libraries Used
| Tool | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models and evaluation metrics |
| Seaborn | Statistical visualisations |
| Matplotlib | Plot rendering and saving |

---

## Files in This Repository
| File | Description |
|---|---|
| income_classification_analysis.py | Complete Python code |
| income.csv | Dataset used for analysis |
| plot1_salary_distribution.png | Salary status distribution |
| plot2_age_distribution.png | Age distribution histogram |
| plot3_age_vs_salstat.png | Age vs salary boxplot |
| plot4_jobtype_vs_salstat.png | JobType vs salary bar chart |
| plot5_education_vs_salstat.png | Education vs salary bar chart |
| plot6_occupation_vs_salstat.png | Occupation vs salary bar chart |
| plot7_capitalgain.png | Capital gain distribution |
| plot8_capitalloss.png | Capital loss distribution |
| plot9_hours_vs_salstat.png | Hours per week vs salary boxplot |
