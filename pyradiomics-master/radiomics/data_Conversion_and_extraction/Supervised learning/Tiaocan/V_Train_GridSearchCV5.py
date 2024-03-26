import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 数据集路径
train_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_0.88_set.csv"
val_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_0.88_set.csv"

# Load datasets
train_set = pd.read_csv(train_set_path)
val_set = pd.read_csv(val_set_path)

# Split features and labels
X_train = train_set.drop(columns=['event', 'time', 'ID'])
y_train = train_set['event']
X_val = val_set.drop(columns=['event', 'time', 'ID'])
y_val = val_set['event']

# Define a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a parameter grid to search
param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1)

# Execute the grid search
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Model evaluation
def evaluate_model(model, X_train, y_train, X_val, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    print("Training set ROC AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    print("Train_utils set ROC AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print("Training set accuracy:", accuracy_score(y_train, y_train_pred))
    print("Train_utils set accuracy:", accuracy_score(y_val, y_val_pred))
    print("Training set recall:", recall_score(y_train, y_train_pred))
    print("Train_utils set recall:", recall_score(y_val, y_val_pred))
    # print("Training set F1 Score:", f1_score(y_train, y_train_pred))
    # print("Train_utils set F1 Score:", f1_score(y_val, y_val_pred))

# Evaluate the best model
evaluate_model(best_model, X_train, y_train, X_val, y_val)
