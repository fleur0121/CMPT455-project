"""Build the reproducible EDA notebook. Run from the repository root."""

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
# Census-Income (KDD): EDA

**結論：年によって記録が異なる列と定数列を除き、33特徴量で比較する。** 以下で欠損・分布・関連を確認する。対象はUCIの2ファイルを結合した18歳以上215,574件。

実行環境：Colab/Kaggle。データがなければ最初のセルで取得する。
"""),
    code("""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

cwd = Path.cwd()
repo_root = cwd if (cwd / "src" / "data_loader.py").exists() else cwd.parent
assert (repo_root / "src" / "data_loader.py").exists(), "リポジトリ直下または notebooks/ で実行してください"
data_path = repo_root / "data" / "processed" / "census_income_adults.csv"
if not data_path.exists():
    subprocess.run([sys.executable, "src/data_loader.py"], cwd=repo_root, check=True)

df = pd.read_csv(data_path, skipinitialspace=True)
assert df.shape == (215_574, 41)
assert "instance_weight" not in df.columns
print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
display(df.head(3))
"""),
    markdown("""
## 1. 欠損値の確認

**方針：`?`のみ欠損。** `Not in universe`は対象外を示すカテゴリとして残す。
"""),
    code("""
# 文字列列の前後の空白を除き、? だけを欠損値にする。
text_cols = df.select_dtypes(include=["object", "str"]).columns
df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
df[text_cols] = df[text_cols].replace("?", pd.NA)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_rate_pct": df.isna().mean().mul(100).round(2),
}).sort_values("missing_count", ascending=False)
display(missing[missing.missing_count > 0])
"""),
    markdown("## 2. `Not in universe` の割合（定数列を確認）"),
    code("""
not_in_universe = pd.DataFrame({
    "count": df.eq("Not in universe").sum(),
    "rate_pct": df.eq("Not in universe").mean().mul(100).round(2),
}).sort_values("count", ascending=False)
display(not_in_universe[not_in_universe["count"] > 0])
"""),
    markdown("## 3. 数値コード5列をカテゴリ型に修正"),
    code("""
# .names では nominal とされるが、CSVから整数として読まれる列。
numeric_categorical = [
    "detailed_industry_recode", "detailed_occupation_recode",
    "own_business_or_self_employed", "veterans_benefits", "year",
]
print("Before:")
display(df[numeric_categorical].dtypes.to_frame("dtype"))
for col in numeric_categorical:
    df[col] = df[col].astype("string")
print("After:")
display(df[numeric_categorical].dtypes.to_frame("dtype"))

numeric_cols = [
    "age", "wage_per_hour", "capital_gains", "capital_losses",
    "dividends_from_stocks", "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
assert set(df.select_dtypes(include="number").columns) == set(numeric_cols)
display(df[numeric_cols].describe(percentiles=[.5, .95, .99]).T)
"""),
    markdown("## 4. 高所得は8.61%（クラス不均衡）"),
    code("""
class_counts = df["income_class"].value_counts(dropna=False)
display(pd.DataFrame({
    "count": class_counts,
    "rate_pct": class_counts.div(len(df)).mul(100).round(2),
}))
"""),
    markdown("""
## 5. 数値7列のクラス別ヒストグラム

**目的：クラス別の分布を比較する。** 金額系の`log1p`は表示用で、データは変えない。
"""),
    code("""
skewed_cols = {"wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks"}
classes = sorted(df["income_class"].dropna().unique())
colors = ["#2878B5", "#D66027"]
fig, axes = plt.subplots(4, 2, figsize=(14, 17))
for ax, col in zip(axes.flat, numeric_cols):
    for label, color in zip(classes, colors):
        values = df.loc[df["income_class"] == label, col].dropna().to_numpy()
        if col in skewed_cols:
            values = np.log1p(values)
        ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.6,
                color=color, label=label)
    ax.set_title(col)
    ax.set_xlabel("log1p(value)" if col in skewed_cols else "value")
    ax.set_ylabel("density within class")
    ax.legend(fontsize=8)
axes.flat[-1].axis("off")
fig.suptitle("Numeric distributions by income class", fontsize=15)
fig.tight_layout()
plt.show()
"""),
    markdown("""
## 6. 数値7列のクラス別箱ひげ図

**目的：分布の中心と広がりを比べる。** 外れ値の点は非表示。金額系の`log1p`は表示用。
"""),
    code("""
fig, axes = plt.subplots(4, 2, figsize=(14, 17))
for ax, col in zip(axes.flat, numeric_cols):
    groups = []
    for label in classes:
        values = df.loc[df["income_class"] == label, col].dropna().to_numpy()
        groups.append(np.log1p(values) if col in skewed_cols else values)
    ax.boxplot(groups, tick_labels=classes, showfliers=False)
    ax.set_title(col)
    ax.set_ylabel("log1p(value)" if col in skewed_cols else "value")
    ax.tick_params(axis="x", labelrotation=15)
axes.flat[-1].axis("off")
fig.suptitle("Numeric boxplots by income class (outlier points hidden)", fontsize=15)
fig.tight_layout()
plt.show()

"""),
    markdown("""
### 外れ値の要約

**結論：金額列の外れ値は一括削除しない。** 0が多く全行のIQRが0になるため、非ゼロ値での外れ値率と、0か否かの高所得率を併記する。99,999が上限コードかは未確認。
"""),
    code("""
def iqr_outlier_rate(values):
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    if iqr == 0:
        return np.nan  # IQRが0になる列ではIQRルールを適用できない
    return ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() * 100

is_high = df["income_class"].eq("50000+.")
outlier_rows = []
for col in numeric_cols:
    values = df[col]
    nonzero = values > 0
    outlier_rows.append({
        "feature": col,
        "zero_rate_pct": (~nonzero).mean() * 100,
        "IQR_outlier_rate_all_pct": iqr_outlier_rate(values),
        "IQR_outlier_rate_nonzero_pct": iqr_outlier_rate(values[nonzero]),
        "max": values.max(),
        "rows_at_max": int(values.eq(values.max()).sum()),
        "high_income_rate_if_nonzero_pct": is_high[nonzero].mean() * 100,
        "high_income_rate_if_zero_pct": is_high[~nonzero].mean() * 100,
    })
display(pd.DataFrame(outlier_rows).set_index("feature").round(2))
"""),
    markdown("## 7. 数値7列の相関（就労関連2列に関連）"),
    code("""
corr = df[numeric_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(9, 7))
heat = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(numeric_cols)), labels=numeric_cols, rotation=55, ha="right")
ax.set_yticks(range(len(numeric_cols)), labels=numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(heat, ax=ax, label="Pearson r")
fig.tight_layout()
plt.show()
"""),
    markdown("""
## 8. カテゴリ列どうしの関連（Cramér's V）

**結論：強い関連があっても即座には列を削除しない。** Cramér's V（0～1）をカテゴリ列・目的変数間で計算する。欠損は`Missing`として数え、定数列は除く。
"""),
    code("""
from scipy.stats.contingency import association

categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "income_class"]
constant_cols = [c for c in categorical_cols if df[c].nunique(dropna=False) == 1]
print("Excluded constant columns:", constant_cols)
cat_frame = df[[c for c in categorical_cols if c not in constant_cols] + ["income_class"]]
codes = {c: pd.factorize(cat_frame[c].fillna("Missing"))[0] for c in cat_frame.columns}

def cramers_v(a, b):
    n_a, n_b = a.max() + 1, b.max() + 1
    table = np.bincount(a * n_b + b, minlength=n_a * n_b).reshape(n_a, n_b)
    return association(table, method="cramer")

names = list(codes)
cramer = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
for i, left in enumerate(names):
    for right in names[i + 1:]:
        cramer.loc[left, right] = cramer.loc[right, left] = cramers_v(codes[left], codes[right])

order = cramer["income_class"].sort_values(ascending=False).index
cramer = cramer.loc[order, order]
fig, ax = plt.subplots(figsize=(15, 13))
heat = ax.imshow(cramer, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(len(order)), labels=order, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(order)), labels=order, fontsize=8)
for i in range(len(order)):
    for j in range(len(order)):
        if i != j and cramer.iloc[i, j] >= 0.5:
            ax.text(j, i, f"{cramer.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white")
fig.colorbar(heat, ax=ax, label="Cramér's V")
ax.set_title("Association between categorical features and income class (V >= 0.5 labelled)")
fig.tight_layout()
plt.show()

pairs = cramer.drop(index="income_class", columns="income_class").stack()
pairs = pairs[[a < b for a, b in pairs.index]].sort_values(ascending=False)
print("Most associated feature pairs:")
display(pairs.head(15).round(3).to_frame("cramers_v"))
print("Association with income_class:")
display(cramer["income_class"].drop("income_class").round(3).to_frame("cramers_v"))
"""),
    markdown("""
## 観察メモ

**結論：33特徴量を基準にし、外れ値は一括削除しない。**

- **除外：** 1995年に全件欠損するmigration 4列、成人では定数の`family_members_under_18`、片方の年で実質的に定数の`live_in_this_house_1_year_ago`と`full_or_part_time_employment_stat`。後者2列は情報のある年の信号を失う代償がある。
- **欠損：** `?`のみ欠損扱い。`Not in universe`はカテゴリとして残す。出生国3列の欠損率は2.09～3.88%。
- **クラス：** 高所得18,566件（8.61%）。層化分割し、主指標はAP。重み付けは訓練CVで判断する。
- **数値：** 金額4列は0が85～97%。`capital_gains`の99,999は578件で、上限コードかは未確認。外れ値削除はせず、raw/`log1p`を比較する。
- **関連：** 就労関連2列のPearson rは約0.66。詳細/大分類の職業・産業列はCramér's V=1.00だが、同一情報とは限らないため削除しない。
- **次段階：** 補完・標準化・one-hot・MIは訓練側だけでfit。PCAは可視化に使用し、最終テストは判断に使わない。
"""),
]

path = Path("notebooks/01_eda.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"Built {path}")
