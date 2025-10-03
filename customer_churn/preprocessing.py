import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


_DATE_REFERENCE = pd.Timestamp("2024-01-01")

_EDUCATION_SYNONYMS = {
    "bachelor": "bachelor",
    "bachelor's": "bachelor",
    "bachelors": "bachelor",
    "diploma": "diploma",
    "high school": "high school",
    "hs": "high school",
    "master": "master",
    "master's": "master",
    "masters": "master",
}

_OCCUPATION_SYNONYMS = {
    "accommodation manager": "accommodation manager",
    "accommodation manager¡": "accommodation manager",
    "accommodation managerá": "accommodation manager",
    "accountant, chartered": "accountant, chartered",
    "accountant, chartered certified": "accountant, chartered certified",
    "accountant, chartered certifie¡": "accountant, chartered certified",
}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert heterogeneous numeric strings to floats."""
    cleaned = series.astype(str).str.replace(r"[^0-9+\-\.e]", "", regex=True)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _standardize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace({"": np.nan, "nan": np.nan})


@dataclass
class ChurnPreprocessor:
    categorical_columns: Iterable[str] = field(
        default_factory=lambda: [
            "gender",
            "marital_status",
            "dependents",
            "occupation",
            "education",
            "segment",
            "preferred_contact",
        ]
    )
    numeric_columns: Iterable[str] = field(
        default_factory=lambda: [
            "income",
            "tenure_years",
            "credit_score",
            "credit_history_years",
            "outstanding_debt",
            "balance",
            "products_count",
            "complaints_count",
            "age",
        ]
    )
    target_column: str = "churned"
    encoders_: Dict[str, LabelEncoder] = field(default_factory=dict, init=False)
    scaler_: Optional[StandardScaler] = field(default=None, init=False)
    numeric_fill_values_: Dict[str, float] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)
    feature_columns_: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.categorical_columns = list(self.categorical_columns)
        self.numeric_columns = list(self.numeric_columns)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        features, _ = self._prepare(df.copy(), fit=True, track_unknowns=False)
        self.fitted_ = True
        return features

    def transform(
        self, df: pd.DataFrame, track_unknowns: bool = False
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        features, summary = self._prepare(
            df.copy(), fit=False, track_unknowns=track_unknowns
        )
        if track_unknowns:
            return features, summary
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.fit(df)
        return features

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ChurnPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, ChurnPreprocessor):
            raise TypeError("Loaded object is not a ChurnPreprocessor instance.")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare(
        self, df: pd.DataFrame, fit: bool, track_unknowns: bool
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df = self._basic_clean(df)
        unknown_tracker: Dict[str, set] = {col: set() for col in self.categorical_columns}

        # Separate target if present
        target = None
        if self.target_column in df.columns:
            target = _to_numeric(df[self.target_column]).fillna(0)
            target = (target > 0).astype(int)
            df = df.drop(columns=[self.target_column])

        # Encode categorical columns
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = _standardize_text(df[col])
            df[col] = df[col].fillna("unknown")
            df[col] = df[col].replace({
                "": "unknown",
                "none": "unknown",
                "na": "unknown",
                "n/a": "unknown",
            })

            if col == "education":
                df[col] = df[col].map(_EDUCATION_SYNONYMS).fillna(df[col])
            if col == "occupation":
                df[col] = df[col].map(_OCCUPATION_SYNONYMS).fillna(df[col])

            if col == "dependents":
                df[col] = df[col].apply(self._normalise_dependents)
            if col == "gender":
                df[col] = df[col].apply(self._normalise_gender)

            if fit:
                classes = sorted(set(df[col].dropna().tolist()) | {"other"})
                encoder = LabelEncoder()
                encoder.fit(classes)
                self.encoders_[col] = encoder
            encoder = self.encoders_[col]

            def _map_value(x: str) -> str:
                if x in encoder.classes_:
                    return x
                if track_unknowns:
                    unknown_tracker[col].add(x)
                return "other"

            df[col] = df[col].apply(_map_value)
            df[col] = encoder.transform(df[col])

        # Numeric columns: ensure existence and numeric type
        for col in self.numeric_columns:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = _to_numeric(df[col])

        if fit:
            self.numeric_fill_values_ = {
                col: df[col].median(skipna=True) for col in self.numeric_columns
            }
            for col, fill_val in self.numeric_fill_values_.items():
                df[col] = df[col].fillna(fill_val)
            self.scaler_ = StandardScaler()
            df[self.numeric_columns] = self.scaler_.fit_transform(df[self.numeric_columns])
            self.feature_columns_ = list(self.categorical_columns) + list(self.numeric_columns)
        else:
            for col, fill_val in self.numeric_fill_values_.items():
                df[col] = df[col].fillna(fill_val)
            df[self.numeric_columns] = self.scaler_.transform(df[self.numeric_columns])

        if target is not None:
            df[self.target_column] = target

        expected_columns: List[str] = list(self.categorical_columns) + list(self.numeric_columns)
        if target is not None:
            expected_columns.append(self.target_column)
        summary = {
            col: sorted(values)
            for col, values in unknown_tracker.items()
            if values
        }

        return df[expected_columns], summary

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r"\s+", "_", col.strip().lower()) for col in df.columns]

        if "age" not in df.columns and "dob" in df.columns:
            df["age"] = df["dob"].apply(self._derive_age)

        drop_cols = {
            "rownumber",
            "customer_id",
            "surname",
            "first_name",
            "address",
            "phone",
            "dob",
            "churn_reason",
            "churn_date",
        }
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

        return df

    @staticmethod
    def _derive_age(value: object) -> float:
        if pd.isna(value):
            return np.nan
        try:
            ts = pd.to_datetime(value, errors="coerce")
        except Exception:
            ts = pd.NaT
        if pd.isna(ts):
            return np.nan
        delta = _DATE_REFERENCE - ts
        return delta.days / 365.25

    @staticmethod
    def _normalise_dependents(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        text = str(value).strip().lower()
        mapping = {
            "yes": "1",
            "y": "1",
            "true": "1",
            "no": "0",
            "n": "0",
            "false": "0",
        }
        if text in mapping:
            return mapping[text]
        text = re.sub(r"[^0-9a-z]+", " ", text).strip()
        if text.isdigit():
            return text
        return text if text else "unknown"

    @staticmethod
    def _normalise_gender(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        text = str(value).strip().lower()
        mapping = {
            "m": "male",
            "male": "male",
            "f": "female",
            "female": "female",
        }
        return mapping.get(text, text if text else "unknown")
