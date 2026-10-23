"""Build the preprocessing comparison notebook from auditable source cells."""

from pathlib import Path

import nbformat as nbf


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook.metadata["language_info"] = {"name": "python"}
notebook.cells = [
    markdown("""
# 手順4・5: 前処理候補の比較

**結論：33特徴量・金額raw・重みなしを後続比較の共通基準にする。** migrationを残す37列、金額`log1p`、重み付けも訓練CVで比較する。行は削除しない。

性能は訓練側30,000件での予備比較。最終テストは使わない。
"""),
    code("""
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

cwd = Path.cwd()
repo_root = cwd if (cwd / "src" / "preprocessing.py").exists() else cwd.parent
assert (repo_root / "src" / "preprocessing.py").exists(), "リポジトリ直下または notebooks/ で実行してください"
sys.path.insert(0, str(repo_root / "src"))
from preprocessing import (
    MIGRATION_COLUMNS, MONEY_COLUMNS, NUMERIC_COLUMNS,
    build_preprocessor, prepare_xy, select_features,
)

data_path = repo_root / "data" / "processed" / "census_income_adults.csv"
if not data_path.exists():
    import subprocess
    subprocess.run([sys.executable, "src/data_loader.py"], cwd=repo_root, check=True)
df = pd.read_csv(data_path, skipinitialspace=True)
X, y = prepare_xy(df)
print(f"Rows: {len(X):,}, original features: {X.shape[1]}, positive rate: {y.mean():.2%}")
"""),
    markdown("""
## 1. 層化80/20分割

**結論：テスト20%は保留。** 補完・変換・MIは訓練側だけでfitする。
"""),
    code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
split_summary = pd.DataFrame({
    "rows": [len(X_train), len(X_test)],
    "positive_count": [int(y_train.sum()), int(y_test.sum())],
    "positive_rate_pct": [round(y_train.mean() * 100, 2), round(y_test.mean() * 100, 2)],
}, index=["train", "test"])
display(split_summary)
assert len(X_train) + len(X_test) == 215_574
"""),
    markdown("""
## 2. 前処理の構成とone-hot後の列数

**方針：数値は中央値補完＋標準化、カテゴリは`Missing`補完＋one-hot。** 未知カテゴリを許容し、fitは訓練側のみ。
"""),
    code("""
feature_counts = []
for include_migration in (False, True):
    X_variant = select_features(X_train, include_migration)
    preprocessor = build_preprocessor(X_variant, log_money=False)
    transformed = preprocessor.fit_transform(X_variant)
    feature_counts.append({
        "migration": "keep" if include_migration else "drop",
        "original_features": X_variant.shape[1],
        "onehot_output_columns": transformed.shape[1],
        "sparse": bool(hasattr(transformed, "tocsr")),
    })
display(pd.DataFrame(feature_counts))
"""),
    markdown("""
## 3. 同じ訓練サンプル・同じ5分割で候補を比較

**目的：同じ30,000件・5分割でAPを比較する。** ロジスティック回帰で前処理を予備比較するため、他の分類器へは一般化しない。
"""),
    code("""
X_dev, _, y_dev, _ = train_test_split(
    X_train, y_train, train_size=30_000, stratify=y_train, random_state=42
)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
for include_migration in (False, True):
    X_variant = select_features(X_dev, include_migration)
    for log_money in (False, True):
        for class_weight in (None, "balanced"):
            model = Pipeline([
                ("preprocess", build_preprocessor(X_variant, log_money)),
                ("classifier", LogisticRegression(
                    solver="liblinear", max_iter=1000, class_weight=class_weight,
                    random_state=42,
                )),
            ])
            scores = cross_validate(
                model, X_variant, y_dev, cv=folds,
                scoring={"AP": "average_precision", "ROC_AUC": "roc_auc"},
                n_jobs=1, error_score="raise",
            )
            results.append({
                "migration": "keep" if include_migration else "drop",
                "money": "log1p" if log_money else "raw",
                "class_weight": class_weight or "none",
                "AP_mean": scores["test_AP"].mean(),
                "AP_sd": scores["test_AP"].std(),
                "ROC_AUC_mean": scores["test_ROC_AUC"].mean(),
                "fit_seconds_mean": scores["fit_time"].mean(),
            })
comparison = pd.DataFrame(results).sort_values("AP_mean", ascending=False)
display(comparison.round(4).reset_index(drop=True))
print(f"Random-ranking AP baseline: {y_dev.mean():.4f}")
"""),
    markdown("""
## 4. 元の特徴量単位で相互情報量を計算

**結論：MIの順位はone-hot前の元33列で付ける。** カテゴリ型を指定し、訓練サンプルだけで計算する。
"""),
    code("""
X_mi = select_features(X_dev, include_migration=False).copy()
cat_cols = [c for c in X_mi.columns if c not in NUMERIC_COLUMNS]
X_mi[NUMERIC_COLUMNS] = SimpleImputer(strategy="median").fit_transform(X_mi[NUMERIC_COLUMNS])
cat_imputed = SimpleImputer(strategy="constant", fill_value="Missing").fit_transform(X_mi[cat_cols])
X_mi[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(cat_imputed)
discrete_mask = [c in cat_cols for c in X_mi.columns]
mi_scores = mutual_info_classif(
    X_mi, y_dev, discrete_features=discrete_mask, random_state=42, n_neighbors=3
)
mi_ranking = pd.DataFrame({"feature": X_mi.columns, "MI_score": mi_scores})
mi_ranking = mi_ranking.sort_values("MI_score", ascending=False).reset_index(drop=True)
display(mi_ranking)
"""),
    markdown("""
## 5. one-hot後の次元とPCA

**結論：PCAは可視化用で、基準パイプラインには入れない。** 30,000件で寄与率とクラス別・年別の散布図を確認する。数値列は標準化、ダミー列は0/1なので寄与の大きさは揃わない。
"""),
    code("""
X_pca_source = select_features(X_dev, include_migration=False)
pca_preprocessor = build_preprocessor(X_pca_source, log_money=False)
X_encoded = pca_preprocessor.fit_transform(X_pca_source).toarray()
pca = PCA(random_state=42).fit(X_encoded)
cumulative = np.cumsum(pca.explained_variance_ratio_)
display(pd.DataFrame({
    "explained_variance": [0.80, 0.90, 0.95, 0.99],
    "components_needed": [int(np.searchsorted(cumulative, t) + 1) for t in (0.80, 0.90, 0.95, 0.99)],
    "encoded_columns": X_encoded.shape[1],
}))

feature_names = pca_preprocessor.get_feature_names_out()
loadings = pd.DataFrame({
    f"PC{i + 1}": pd.Series(pca.components_[i], index=feature_names)
    .abs().sort_values(ascending=False).head(5).index
    for i in range(2)
})
print("Largest absolute loadings:")
display(loadings)

components = pca.transform(X_encoded)[:, :2]
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, color="#2878B5")
for t in (0.80, 0.90, 0.95):
    axes[0].axhline(t, color="grey", linestyle=":", linewidth=1)
axes[0].set_xlabel("number of principal components")
axes[0].set_ylabel("cumulative explained variance")
axes[0].set_title("PCA on one-hot encoded training sample")
for label, name, color in ((0, "- 50000.", "#2878B5"), (1, "50000+.", "#D66027")):
    mask = y_dev.to_numpy() == label
    axes[1].scatter(components[mask, 0], components[mask, 1], s=4, alpha=0.35,
                    color=color, label=f"{name} (n={mask.sum():,})")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].set_title("First two principal components by income class")
axes[1].legend(markerscale=4)
year_values = X_pca_source["year"].to_numpy()
for year, color in (("94", "#4C9F70"), ("95", "#8E5EA2")):
    mask = year_values == year
    axes[2].scatter(components[mask, 0], components[mask, 1], s=4, alpha=0.35,
                    color=color, label=f"19{year} (n={mask.sum():,})")
axes[2].set_xlabel(axes[1].get_xlabel())
axes[2].set_ylabel(axes[1].get_ylabel())
axes[2].set_title("First two principal components by survey year")
axes[2].legend(markerscale=4)
fig.tight_layout()
plt.show()
"""),
    markdown("""
## 判断メモ

**結論：33特徴量・raw・重みなしを共通基準にする。**

- **除外：** migration 4列は1995年に全件欠損。保持/除外のLR AP差は0.0004で、性能差ではなく年依存の欠損が理由。定数列と年依存2列も除く。
- **金額：** LRはraw 0.6048、`log1p` 0.5927。rawを基準とする。他の分類器での影響は分類の段階で確認する。
- **重み：** LRはなし0.6048、`balanced` 0.5975。なしを基準とするが、他モデルへ一般化しない。
- **次元：** one-hot後は訓練全体441列、30,000件では440列。PCAの90%寄与率に63成分。年別の帯は2列除外後に見えなくなった。PCAは基準パイプラインには使わない。

すべて訓練側の予備結果。後続CVでも前処理は各訓練foldでfitする。
"""),
]

path = Path("notebooks/02_preprocessing.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"Built {path}")
