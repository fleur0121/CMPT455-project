"""Load and prepare the UCI Census-Income (KDD) data for this project.

The raw files are intentionally not committed to Git.  Running this module
downloads them into ``data/raw`` and writes the cleaned, age-filtered data to
``data/processed/census_income_adults.csv``.
"""

from pathlib import Path
import io
import tarfile
import zipfile
from urllib.request import Request, urlopen

import pandas as pd


DATASET_URL = "https://archive.ics.uci.edu/static/public/117/census%2Bincome%2Bkdd.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

COLUMNS = [
    "age", "class_of_worker", "detailed_industry_recode",
    "detailed_occupation_recode", "education", "wage_per_hour",
    "enroll_in_edu_inst_last_wk", "marital_stat", "major_industry_code",
    "major_occupation_code", "race", "hispanic_origin", "sex",
    "member_of_labor_union", "reason_for_unemployment",
    "full_or_part_time_employment_stat", "capital_gains", "capital_losses",
    "dividends_from_stocks", "tax_filer_stat", "region_of_previous_residence",
    "state_of_previous_residence", "detailed_household_and_family_stat",
    "detailed_household_summary_in_household", "instance_weight",
    "migration_code_change_in_msa", "migration_code_change_in_reg",
    "migration_code_move_within_reg", "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
    "family_members_under_18", "country_of_birth_father",
    "country_of_birth_mother", "country_of_birth_self", "citizenship",
    "own_business_or_self_employed", "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits", "weeks_worked_in_year", "year", "income_class",
]

RAW_FILES = ("census-income.data", "census-income.test")


def download_raw_files() -> None:
    """Download the two source files if they are not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "census-income-kdd.zip"
    if not archive_path.exists():
        print("Downloading the UCI Census-Income archive ...")
        request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            archive_path.write_bytes(response.read())

    with zipfile.ZipFile(archive_path) as archive:
        tar_bytes = archive.read("census.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
        for filename in RAW_FILES:
            destination = RAW_DIR / filename
            if destination.exists():
                continue
            print(f"Extracting {filename} ...")
            destination.write_bytes(archive.extractfile(filename).read())

def load_data(download: bool = True) -> pd.DataFrame:
    """Read, concatenate, filter to adults, and remove instance weight."""
    if download:
        download_raw_files()

    frames = [
        pd.read_csv(RAW_DIR / filename, header=None, names=COLUMNS, skipinitialspace=True)
        for filename in RAW_FILES
    ]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined rows: {len(combined):,}")
    assert len(combined) == 299_285, "Unexpected combined row count"

    adults = combined.loc[combined["age"] >= 18].drop(columns="instance_weight").copy()
    print(f"Rows after age >= 18: {len(adults):,}")
    assert len(adults) == 215_574, "Unexpected adult row count"
    return adults


if __name__ == "__main__":
    data = load_data()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "census_income_adults.csv"
    data.to_csv(output, index=False)
    print(f"Saved: {output} ({len(data):,} rows, {len(data.columns)} columns)")
