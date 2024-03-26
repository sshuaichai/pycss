from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import gc

# 创建包含多个模型实例的字典
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(solver='saga', max_iter=10000, tol=1e-2),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0, iterations=100),
    "LGBMClassifier": LGBMClassifier(min_child_samples=20),
    # 假设LassoCV已被正确初始化，这里仅作为示例
    "Lasso": LassoCV(cv=10, random_state=42)
}

# 如果你想删除"Random Forest"和"Logistic Regression"模型实例
del models['Random Forest']
del models['Logistic Regression']

# 删除整个字典
del models

# 强制执行垃圾回收
gc.collect()

# 查看当前未被回收的对象
for obj in gc.get_objects():
    if isinstance(obj, dict):
        print(obj)  # 仅打印字典类型的对象，减少输出

if 'models' in locals():
    print("models still exists in locals.")
else:
    print("models does not exist in locals.")

if 'models' in globals():
    print("models still exists in globals.")
else:
    print("models does not exist in globals.")

# 如果你想要重置为一个空字典，而不是删除它，可以取消注释下面的代码
# models = {}
