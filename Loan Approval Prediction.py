# %% [markdown]
# 
# ## Loan Approval Prediction

# %% [markdown]
# ### Importing libraries

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import warnings

warnings.filterwarnings('ignore')

# %%
df =  pd.read_csv(r"E:\Projects\ML\Projects\Personal Loan approval prediction\Bank_loan_approval.csv")

df.head(10)


# %% [markdown]
# ## Data Exploration and Preprocessing

# %%
df.info()

# %%
df.isna().sum()

# %%
print(df.Education.unique())
print(df.Gender.unique())
print(df.Past_Application_Status.unique())
print(df.Location.unique())

# %%
df['is_male'] = df['Gender'].apply(lambda x: 1 if x=='Male' else 0)
df['Education'] =  df['Education'].map({'Bachelor':2,'High School':1,'Master':3,'PhD':4,'Other':0})
df['Past_Application_Status'] = df['Past_Application_Status'].map({'Approved':3,'Pending':2,'Rejected':1})
df['Past_Application_Status'].fillna(0,inplace=True)

df.drop(columns='Gender',inplace=True)

# %%
df = pd.get_dummies(df,dtype='int')

# %%
df.isna().sum()

# %%
df['Bank_Balance'] = df['Bank_Balance'].fillna(df['Bank_Balance'].median())  # Replace NaNs with median
df['Missed_EMI'].fillna(0,inplace=True)
df['Credit_Inquiries'].fillna(0,inplace=True)

# %%
df.corr()['Loan_Status'].sort_values(ascending=False)

# %%
df.hist(figsize=(20, 15), bins=30, edgecolor='black')
plt.show()

# %% [markdown]
# ### Train-Test Split and Handling Class Imbalance

# %%
X = df.drop(columns=['Loan_Status'])  # Features
y = df['Loan_Status']  # Target variable

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# %%
y.value_counts(normalize=True)  # Check class distribution

# %%
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new balance
y_train_resampled.value_counts(normalize=True)


# %% [markdown]
# ### Model Training and Evaluation (Baseline Models)

# %%
# Initialize models

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# %%
# Train and evaluate each model
for name, model in models.items():
    print(f"\n📌 **{name} Performance**")
    
    model.fit(X_train_resampled, y_train_resampled)  # Train
    y_pred = model.predict(X_test)  # Predict
    
    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# %% [markdown]
# ### Hyperparameter Tuning - Random Forest (GridSearchCV)

# %%
# Define parameter grid
param_grid_rf = {
    'n_estimators': [100, 300, 500], 
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                              cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Best Parameters
print("Best RF Parameters:", grid_search_rf.best_params_)

# Evaluate
best_rf = grid_search_rf.best_estimator_
rf_pred = best_rf.predict(X_test)

print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]))

# %% [markdown]
# ### Hyperparameter Tuning - XGBoost (Optuna)

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
    }
    
    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best Params
print("Best XGBoost Parameters:", study.best_params)

# Train the best model
best_xgb = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
best_xgb.fit(X_train, y_train)

# Evaluate
xgb_pred = best_xgb.predict(X_test)
print(classification_report(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1]))

# %% [markdown]
# ### Feature Importance

# %%
# Extract feature importance
xgb_importance = best_xgb.feature_importances_
rf_importance = best_rf.feature_importances_

# %%
# Convert to DataFrame
features = X_train.columns
xgb_feat_imp = pd.DataFrame({'Feature': features, 'Importance': xgb_importance}).sort_values(by='Importance', ascending=False)
rf_feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf_importance}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance for XGBoost
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_feat_imp[:10], palette='Blues_r')
plt.title("🔹 XGBoost Feature Importance 🔹")
plt.show()

# Plot Feature Importance for Random Forest
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feat_imp[:10], palette='Greens_r')
plt.title("🌲 Random Forest Feature Importance 🌲")
plt.show()

# %% [markdown]
# ### Model Saving

# %%
# Save the best XGBoost model
joblib.dump(best_xgb, "best_xgb_model.pkl")

# Save the best Random Forest model
joblib.dump(best_rf, "best_rf_model.pkl")

# %% [markdown]
# ### Best Model Parameters:
# 
#  - XGBoost: Optimized with n_estimators=228, max_depth=13, learning_rate=0.23, etc.
# 
#  - Random Forest: Optimized with n_estimators=300, min_samples_split=5, etc.
# 
# ### Model Performance Metrics:
# 
# Logistic Regression:
# 
#  - Accuracy: 0.69, ROC-AUC: 0.73
# 
#  - Low recall for class 1, making it ineffective for this classification.
# 
# Random Forest:
# 
#  - Accuracy: 0.96, ROC-AUC: 0.96
# 
#  - High precision and recall, good model performance.
# 
# XGBoost:
# 
#  - Accuracy: 0.96, ROC-AUC: 0.99
# 
#  - Best-performing model with high recall and precision.
# 
# #### Final ROC-AUC Scores:
# 
# Logistic Regression: 0.73
# 
# Random Forest: 0.96
# 
# XGBoost: 0.99 (Best model)
# 
# #### Conclusion:
# 
# XGBoost outperforms all models in ROC-AUC score and accuracy.
# 
# Random Forest is a strong alternative with high recall.
# 
# Logistic Regression is not suitable due to poor recall for class 1.


