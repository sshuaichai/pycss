import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

# 1. 移除高度相关的特征
def remove_highly_correlated_features(df, threshold=0.9):  #阈值
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

# 2. 保存特征相关性的热图
def save_heatmap(df, output_folder, filename="feature_heatmap.tiff"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', xticklabels=False, yticklabels=False, cbar_kws={'shrink': .5})
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, format='tiff', dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_path}")

# 3. 基于模型使用ANOVA选择特征
def find_best_k(X, y, min_k=1, max_k=None, cv=10): #交叉
    if max_k is None:
        max_k = X.shape[1]
    scores = []
    k_values = range(min_k, max_k + 1)
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for k in k_values:
        selector = SelectKBest(f_classif, k=k)
        model = LGBMClassifier(n_estimators=100, random_state=42)
        pipeline = Pipeline([('selector', selector), ('model', model)])
        score = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc').mean()
        scores.append(score)
    best_score = max(scores)
    best_k = k_values[scores.index(best_score)]
    print(f"Best k: {best_k} with score: {best_score}")
    return best_k, best_score

# 预处理和特征选择
def preprocess_and_select_features_with_best_k(data_path, output_folder):
    df = pd.read_csv(data_path)
    y = df.pop('event')
    time = df.pop('time')
    ID = df.pop('ID')
    X = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_filtered, dropped_features = remove_highly_correlated_features(X_scaled, threshold=0.9)
    best_k, _ = find_best_k(X_filtered, y)
    selector = SelectKBest(f_classif, k=best_k)
    X_selected = selector.fit_transform(X_filtered, y)
    selected_features = X_filtered.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    X_selected_df['time'] = time.values
    X_selected_df['event'] = y.values
    X_selected_df['ID'] = ID.values
    selected_features_file = os.path.join(output_folder, "selected_features_with_bestk_LGBM.csv")
    X_selected_df.to_csv(selected_features_file, index=False)
    print(f"Selected features with best k saved to {selected_features_file}")
    save_heatmap(X_selected_df.drop(['event', 'time', 'ID'], axis=1), output_folder)

# 调用示例
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_ID_updated.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
preprocess_and_select_features_with_best_k(data_path, output_folder)
