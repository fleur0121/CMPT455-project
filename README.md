# CMPT455-project

## Step 2: Data loading

From the project root, run:

```bash
python src/data_loader.py
```

The script downloads `census-income.data` and `census-income.test` from the
UCI Census-Income (KDD) database, assigns the 42 documented column names,
combines the files, keeps adults (`age >= 18`), removes `instance_weight`,
and verifies the expected 215,574 rows. Raw and generated data are ignored by
Git; the download can be repeated in Colab or Kaggle.

## Step 3: EDA

Open `notebooks/01_eda.ipynb` in Colab, Kaggle, or Jupyter and run all cells.
The notebook downloads the source data automatically if the processed CSV is
absent, then checks missing values, categorical codes, class balance, numeric
distributions, outliers (including a zero-inflation summary for money
columns), numeric correlations, and Cramér's V between categorical features.
Keep the executed cell outputs when submitting the notebook.

## Step 4–5: Preprocessing candidates

Open `notebooks/02_preprocessing.ipynb` and run all cells. It compares keeping
versus dropping four migration columns (after removing the adult-only constant
column and two columns that carry information in only one survey year:
`live_in_this_house_1_year_ago` and `full_or_part_time_employment_stat`), raw versus `log1p` money features, and
class weighting. It uses a stratified 80/20 split, then the same 30,000-row
training subsample and 5-fold splits for exploratory comparisons. The test
set remains untouched. It also reports one-hot column counts and ranks the
33 baseline original features by mutual information on training data. Finally,
it runs PCA on the one-hot encoded training sample and plots the cumulative
explained variance and the first two components by income class and by
survey year.
