import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    y = df.pop('event')
    df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    corr_matrix = pd.DataFrame(X_scaled, columns=df.columns).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    X_deduplicated = pd.DataFrame(X_scaled, columns=df.columns).drop(columns=to_drop)
    return X_deduplicated, y

def select_and_evaluate_features(X, y, output_folder):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # 使用SelectKBest
    best_k = None
    best_score = 0
    for k in range(1, X.shape[1] + 1):
        selector = SelectKBest(f_classif, k=k) # 执行方差分析（ANOVA）
        pipeline = make_pipeline(selector, LogisticRegression(max_iter=1000, random_state=42))
        score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
        if score > best_score:
            best_score = score
            best_k = k
    results['SelectKBest'] = best_score

    # RFE方法
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    score = cross_val_score(selector, X, y, cv=cv, scoring='accuracy').mean()
    results['RFE'] = score

    # PCA方法
    pca = PCA(n_components=10)
    pipeline = Pipeline([('PCA', pca), ('Model', LogisticRegression(max_iter=1000, random_state=42))])
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
    results['PCA'] = score

    # L1正则化方法
    estimator = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    score = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
    results['L1'] = score

    # 比较不同方法的性能
    best_method = max(results, key=results.get)
    print(f"Best feature selection method: {best_method} with CV accuracy: {results[best_method]}")
    print("All methods' scores:", results)

    return best_k, best_method

def main(data_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    X, y = preprocess_data(data_path)
    best_k, best_method = select_and_evaluate_features(X, y, output_folder)
    # 注意：这里需要根据所选方法进行后续处理

if __name__ == "__main__":
    DATA_PATH = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_RB_ID_updated.csv"
    OUTPUT_FOLDER = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
    main(DATA_PATH, OUTPUT_FOLDER)
