"""
OULAD data loader for standalone prediction experiments.

Turns raw OULAD CSVs into clean parquet artifacts:
- daily VLE click aggregation (enrollment x date x activity_type)
- student-level target table (final_result + demographics + deadlines)
- presentation-level train/val/test splits

Writes processed tables and fixed train/validation/test splits.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ENROLL_KEYS = ["id_student", "code_module", "code_presentation"]
PRES_KEYS = ["code_module", "code_presentation"]
TRAIN_PRESENTATIONS = {"2013J", "2013B"}
TEST_PRESENTATIONS = {"2014J", "2014B"}

# studentVle is ~453MB / ~10.6M rows; explicit dtypes keep peak RAM down.
# ponytail: pandas in-RAM (~3GB peak); switch to chunked read or duckdb if OOM.
VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int32",
    "sum_click": "int32",
}


class OULADLoader:
    """Load and transform raw OULAD CSVs into split parquet artifacts."""

    def __init__(self, data_dir: str | Path, out_dir: str | Path = "data/oulad"):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)

    def load_target(self) -> pd.DataFrame:
        """Build one-row-per-enrollment target table with demographics + deadlines."""
        required = ["studentInfo.csv", "studentRegistration.csv", "courses.csv", "assessments.csv"]
        missing = [name for name in required if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "OULAD raw CSV files were not found in "
                f"{self.data_dir.resolve()}. Missing: {', '.join(missing)}. "
                "The existing data/oulad/processed files can be used without --build; "
                "download/extract the official OULAD CSV archive to run a full rebuild."
            )
        info = pd.read_csv(self.data_dir / "studentInfo.csv")
        reg = pd.read_csv(self.data_dir / "studentRegistration.csv")
        courses = pd.read_csv(self.data_dir / "courses.csv")
        assess = pd.read_csv(self.data_dir / "assessments.csv")

        # first deadline = earliest non-null assessment date per presentation
        first_deadline = (
            assess.dropna(subset=["date"])
            .groupby(PRES_KEYS)["date"]
            .min()
            .rename("first_deadline")
            .reset_index()
        )

        df = info.merge(reg, on=ENROLL_KEYS, how="left")
        df = df.merge(courses, on=PRES_KEYS, how="left")
        df = df.merge(first_deadline, on=PRES_KEYS, how="left")
        df["first_deadline"] = df["first_deadline"].fillna(df["module_presentation_length"])
        logger.info("load_target: %d enrollments", len(df))
        return df

    def load_daily_vle(self) -> pd.DataFrame:
        """Aggregate studentVle clicks per (enrollment, date, activity_type)."""
        svle = pd.read_csv(self.data_dir / "studentVle.csv", dtype=VLE_DTYPES)
        vle = pd.read_csv(
            self.data_dir / "vle.csv",
            usecols=["id_site", "code_module", "code_presentation", "activity_type"],
            dtype={"id_site": "int32", "code_module": "category", "code_presentation": "category"},
        )

        merged = svle.merge(vle, on=["id_site", "code_module", "code_presentation"],
                            how="left")
        n_total = len(merged)
        n_missing = int(merged["activity_type"].isna().sum())
        if n_total:
            logger.info("load_daily_vle: %d/%d rows have no vle match (%.3f%%)",
                        n_missing, n_total, 100 * n_missing / n_total)
        merged = merged.dropna(subset=["activity_type"])

        daily = (
            merged.groupby(ENROLL_KEYS + ["date", "activity_type"], observed=True)[
                "sum_click"
            ]
            .sum()
            .reset_index()
        )
        logger.info("load_daily_vle: %d daily rows", len(daily))
        return daily

    def filter_active(
        self,
        target: pd.DataFrame,
        daily_vle: pd.DataFrame,
        min_days: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Keep enrollments with >= min_days distinct VLE days before first_deadline."""
        dl = target[ENROLL_KEYS + ["first_deadline"]]
        merged = daily_vle.merge(dl, on=ENROLL_KEYS, how="left")
        pre = merged[merged["date"] < merged["first_deadline"]]
        active = (
            pre.groupby(ENROLL_KEYS)["date"]
            .nunique()
            .rename("active_days")
            .reset_index()
        )
        keep = active.loc[active["active_days"] >= min_days, ENROLL_KEYS]
        target_f = target.merge(keep, on=ENROLL_KEYS, how="inner")
        daily_f = daily_vle.merge(keep, on=ENROLL_KEYS, how="inner")
        logger.info("filter_active: kept %d/%d enrollments (min_days=%d)",
                    len(target_f), len(target), min_days)
        return target_f, daily_f

    def make_splits(
        self,
        target: pd.DataFrame,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Presentation-level split; val carved from train, stratified by final_result."""
        cols = ENROLL_KEYS + ["final_result"]
        train_full = target[target["code_presentation"].isin(TRAIN_PRESENTATIONS)]
        test = target[target["code_presentation"].isin(TEST_PRESENTATIONS)]

        try:
            train_idx, val_idx = train_test_split(
                train_full.index,
                test_size=val_frac,
                stratify=train_full["final_result"],
                random_state=seed,
            )
        except ValueError:
            logger.warning(
                "make_splits: stratified split failed (a class has <2 members); "
                "retrying with stratify=None"
            )
            train_idx, val_idx = train_test_split(
                train_full.index,
                test_size=val_frac,
                stratify=None,
                random_state=seed,
            )
        splits = {
            "train": train_full.loc[train_idx, cols].reset_index(drop=True),
            "val": train_full.loc[val_idx, cols].reset_index(drop=True),
            "test": test[cols].reset_index(drop=True),
        }
        logger.info("make_splits: train=%d val=%d test=%d",
                    len(splits["train"]), len(splits["val"]), len(splits["test"]))
        return splits

    def build(self) -> None:
        """Run the full pipeline and write parquet artifacts."""
        target = self.load_target()
        daily = self.load_daily_vle()
        target, daily = self.filter_active(target, daily)
        splits = self.make_splits(target)

        proc = self.out_dir / "processed"
        sp = self.out_dir / "splits"
        proc.mkdir(parents=True, exist_ok=True)
        sp.mkdir(parents=True, exist_ok=True)

        daily.to_parquet(proc / "daily_vle.parquet", index=False)
        target.to_parquet(proc / "student_target.parquet", index=False)
        for name, df in splits.items():
            df.to_parquet(sp / f"{name}.parquet", index=False)
        logger.info("build: wrote artifacts to %s", self.out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build OULAD parquet artifacts")
    parser.add_argument("--data-dir", default="../../датасет/oulad",
                        help="Directory with raw OULAD CSVs")
    parser.add_argument("--out-dir", default="data/oulad",
                        help="Output directory for parquet artifacts")
    parser.add_argument("--build", action="store_true", help="Run the full build")
    args = parser.parse_args()

    loader = OULADLoader(data_dir=args.data_dir, out_dir=args.out_dir)
    if args.build:
        loader.build()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
