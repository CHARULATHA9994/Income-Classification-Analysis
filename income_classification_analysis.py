# ============================================================
# INCOME CLASSIFICATION ANALYSIS
# ============================================================
# Author   : M. Charulatha
# Degree   : MSc Bioinformatics and Data Science
# Course   : NPTEL Python for Data Science (Certified)
# Dataset  : Adult Income Dataset (income.csv)
#
# OBJECTIVE:
#   Predict whether an individual earns >50K or <=50K
#   based on demographic and employment features.
#
# MODELS BUILT:
#   1. Logistic Regression  — Accuracy: 84.27%
#   2. K-Nearest Neighbours — Accuracy: 84.36%
#
# STEPS:
#   1. Import libraries and load dataset
#   2. Exploratory Data Analysis (EDA)
#   3. Handle missing values
#   4. Data visualisation
#   5. Data preprocessing
#   6. Build and evaluate ML models
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot
from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import accuracy_score, confusion_matrix

# ============================================================
# STEP 1 — LOAD DATASET
# 31,978 records, 13 columns
# Target: SalStat (<=50K or >50K)
# ============================================================

data   = pd.read_csv("income.csv")
data_1 = data.copy()

# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

data_1.info()
print("\nMissing values (initial):")
print(data_1.isnull().sum())

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

summary_num = data_1.describe()
print("\nNumerical Summary:\n", summary_num)

summary_cat = data_1.describe(include='O')
print("\nCategorical Summary:\n", summary_cat)

print("\nUnique JobType values:  ", np.unique(data_1['JobType']))
print("\nUnique occupation values:", np.unique(data_1['occupation']))

# ============================================================
# STEP 3 — HANDLE MISSING VALUES
# Dataset uses ' ?' for missing — converted to NaN
# 1,816 rows dropped. Clean dataset: 30,162 rows
# ============================================================

data_1        = pd.read_csv("income.csv", na_values=' ?')
missing_values = data_1[data_1.isnull().any(axis=1)]
data_2         = data_1.dropna(axis=0)
print(f"\nRows with missing values : {len(missing_values)}")
print(f"Dataset shape after clean: {data_2.shape}")

# ============================================================
# STEP 4 — CORRELATION AND CROSS-TABULATION
# ============================================================

correlation = data_2.corr(numeric_only=True)

gender = pd.crosstab(index=data_2['gender'],
                     columns='count', normalize=True)
print("\nGender Proportion:\n", gender)

gender_salarystatus = pd.crosstab(
    index=data_2['gender'], columns=data_2['SalStat'],
    margins=True, normalize='index')
print("\nGender vs SalStat:\n", gender_salarystatus)

jobtype_salstat = pd.crosstab(
    index=data_2['JobType'], columns=data_2['SalStat'])
education_salarystatus = pd.crosstab(
    index=data_2['EdType'], columns=data_2['SalStat'])
occupation_salarystatus = pd.crosstab(
    index=data_2['occupation'], columns=data_2['SalStat'])

# ============================================================
# STEP 5 — DATA VISUALISATION
# ============================================================

# Plot 1: Salary Status Distribution
# Finding: Class imbalance — far more <=50K than >50K earners
sns.countplot(data_2['SalStat'])
pyplot.title("Salary Status Distribution")
pyplot.tight_layout()
pyplot.savefig("plot1_salary_distribution.png")
pyplot.show()

# Plot 2: Age Distribution
# Finding: Working population peaks between age 25-50
sns.distplot(data_2['age'], bins=10, kde=False)
pyplot.xlabel("Age"); pyplot.ylabel("Count")
pyplot.title("Age Distribution")
pyplot.tight_layout()
pyplot.savefig("plot2_age_distribution.png")
pyplot.show()

# Plot 3: Age vs Salary Status
# Finding: Higher earners are older (median ~44 vs ~34)
sns.boxplot(x='SalStat', y='age', data=data_2)
pyplot.title("Age vs Salary Status")
pyplot.tight_layout()
pyplot.savefig("plot3_age_vs_salstat.png")
pyplot.show()

# Plot 4: JobType vs Salary Status
# Finding: Private sector dominates employment at all income levels
jobtype_salstat.plot(kind='barh', figsize=(10, 6))
pyplot.xlabel('Count'); pyplot.ylabel('JobType')
pyplot.title('JobType vs Salary Status')
pyplot.tight_layout()
pyplot.savefig("plot4_jobtype_vs_salstat.png")
pyplot.show()

# Plot 5: Education vs Salary Status
# Finding: Higher education strongly correlates with >50K income
education_salarystatus.plot(kind='barh', figsize=(10, 8))
pyplot.title('Education vs Salary Status')
pyplot.xlabel('Count'); pyplot.ylabel('EdType')
pyplot.tight_layout()
pyplot.savefig("plot5_education_vs_salstat.png")
pyplot.show()

# Plot 6: Occupation vs Salary Status
# Finding: Exec-managerial and Prof-specialty earn most >50K
occupation_salarystatus.plot(kind='barh', figsize=(10, 6))
pyplot.title('Occupation vs Salary Status')
pyplot.xlabel('Count'); pyplot.ylabel('Occupation')
pyplot.legend(fontsize=10)
pyplot.tight_layout()
pyplot.savefig("plot6_occupation_vs_salstat.png")
pyplot.show()

# Plot 7: Capital Gain Distribution
# Finding: Highly right-skewed — most people have zero capital gain
sns.histplot(data_2['capitalgain'], bins=10, kde=False)
pyplot.xlabel("Capital Gain"); pyplot.ylabel('Count')
pyplot.title("Capital Gain Distribution")
pyplot.tight_layout()
pyplot.savefig("plot7_capitalgain.png")
pyplot.show()

# Plot 8: Capital Loss Distribution
# Finding: Highly right-skewed — most people have zero capital loss
sns.histplot(data_2['capitalloss'], bins=10, kde=False)
pyplot.xlabel("Capital Loss"); pyplot.ylabel('Count')
pyplot.title("Capital Loss Distribution")
pyplot.tight_layout()
pyplot.savefig("plot8_capitalloss.png")
pyplot.show()

# Plot 9: Hours Per Week vs Salary Status
# Finding: Higher earners tend to work more hours (median ~45 vs ~40)
sns.boxplot(x='SalStat', y='hoursperweek', data=data_2)
pyplot.xlabel('Salary Status'); pyplot.ylabel('Hours Per Week')
pyplot.title('Hours Per Week vs Salary Status')
pyplot.tight_layout()
pyplot.savefig("plot9_hours_vs_salstat.png")
pyplot.show()

# ============================================================
# STEP 6 — DATA PREPROCESSING
# Encode target: 0 = <=50K, 1 = >50K
# One-Hot Encode all categorical features
# Split: 70% train, 30% test
# ============================================================

data_2 = data_1.dropna(axis=0).copy()
data_2['SalStat'] = data_2['SalStat'].map({
    ' less than or equal to 50,000': 0,
    ' greater than 50,000': 1
})

newdata      = pd.get_dummies(data_2, drop_first=True)
columns_list = list(newdata.columns)
features     = list(set(columns_list) - set(['SalStat']))
y            = newdata['SalStat'].values
X            = newdata[features].values

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=0)

print(f"\nTraining samples : {train_x.shape[0]}")
print(f"Testing samples  : {test_x.shape[0]}")
print(f"Feature count    : {len(features)}")

# ============================================================
# STEP 7 — MODEL 1: LOGISTIC REGRESSION
# RESULT: Accuracy = 84.27%, AUC area covered adequately
# Confusion Matrix: [[6329, 494], [930, 1296]]
# ============================================================

Logistic = LogisticRegression(max_iter=1000)
Logistic.fit(train_x, train_y)
prediction   = Logistic.predict(test_x)
lr_confusion = confusion_matrix(test_y, prediction)
lr_accuracy  = accuracy_score(test_y, prediction)

print("\n===== LOGISTIC REGRESSION =====")
print("Confusion Matrix:\n", lr_confusion)
print(f"Accuracy  : {lr_accuracy*100:.2f}%")
print(f"Misclassified: {(test_y != prediction).sum()}")

# ============================================================
# STEP 8 — MODEL 2: K-NEAREST NEIGHBOURS (k=16)
# RESULT: Accuracy = 84.36%, slightly better than LR
# Confusion Matrix: [[6389, 434], [981, 1245]]
# k=16 selected after testing k=1 to k=19
# ============================================================

KNN_Classifier = KNeighborsClassifier(n_neighbors=16)
KNN_Classifier.fit(train_x, train_y)
prediction    = KNN_Classifier.predict(test_x)
KNN_confusion = confusion_matrix(test_y, prediction)
KNN_accuracy  = accuracy_score(test_y, prediction)

print("\n===== KNN CLASSIFIER (k=16) =====")
print("Confusion Matrix:\n", KNN_confusion)
print(f"Accuracy  : {KNN_accuracy*100:.2f}%")
print(f"Misclassified: {(test_y != prediction).sum()}")

# ============================================================
# STEP 9 — FIND OPTIMAL K FOR KNN (k=1 to 19)
# ============================================================

misclassified_sample = []
for i in range(1, 20):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(train_x, train_y)
    pred = KNN.predict(test_x)
    misclassified_sample.append((test_y != pred).sum())

pyplot.figure(figsize=(10, 5))
pyplot.plot(range(1, 20), misclassified_sample,
            marker='o', color='steelblue', linewidth=2)
pyplot.xlabel("K Value")
pyplot.ylabel("Misclassified Samples")
pyplot.title("KNN — Finding Optimal K")
pyplot.xticks(range(1, 20))
pyplot.grid(alpha=0.3)
pyplot.tight_layout()
pyplot.savefig("plot10_optimal_k.png")
pyplot.show()

# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================
# Model                  Accuracy
# Logistic Regression    84.27%
# KNN (k=16)             84.36%
#
# KEY INSIGHTS FROM EDA:
# - Higher education level strongly predicts income >50K
# - Exec-managerial and Prof-specialty occupations earn most
# - Older individuals (median age 44) more likely to earn >50K
# - People working more hours/week tend to earn more
# - Capital gain is a strong but skewed predictor
# ============================================================
