"""
Version configuration for the chess move classifier pipeline.

Each VersionConfig instance encapsulates all paths, labels, and flags for a
specific feature-engineering version (V1, V2, V3).  Notebooks import the
pre-built V1/V2/V3 objects instead of using if/else flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib

DATA_DIR = Path("data")


@dataclass
class VersionConfig:
    version: int
    label: str
    n_features: int
    suffix: str
    v2_flag: bool
    v3_flag: bool
    prior_versions: list[VersionConfig] = field(default_factory=list)
    features_suffix: str | None = None

    @property
    def features_csv(self) -> Path:
        s = self.features_suffix if self.features_suffix is not None else self.suffix
        return DATA_DIR / "features" / f"features{s}.csv"

    @property
    def models_dir(self) -> Path:
        return DATA_DIR / f"models{self.suffix}"

    @property
    def eval_dir(self) -> Path:
        return DATA_DIR / f"evaluation{self.suffix}"

    @property
    def rules_suffix(self) -> str:
        return self.suffix

    def load_models(self) -> tuple:
        """Return (decision_tree, random_forest, feature_names) or
        (decision_tree, random_forest, xgboost, feature_names) for V4+."""
        with open(self.models_dir / "feature_names.json") as f:
            feature_names = json.load(f)
        dt = joblib.load(self.models_dir / "decision_tree.joblib")
        rf = joblib.load(self.models_dir / "random_forest.joblib")
        xgb_path = self.models_dir / "xgboost.joblib"
        if xgb_path.exists():
            xgb = joblib.load(xgb_path)
            return dt, rf, xgb, feature_names
        return dt, rf, feature_names

    def load_thresholds(self) -> dict:
        """Return threshold dict if available, else empty dict."""
        path = self.models_dir / "thresholds.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def __repr__(self) -> str:
        return f"V{self.version}({self.n_features} features)"


V1 = VersionConfig(
    version=1,
    label="V1 — posicional (33 features)",
    n_features=33,
    suffix="",
    v2_flag=False,
    v3_flag=False,
)

V2 = VersionConfig(
    version=2,
    label="V2 — posicional + tática (52 features)",
    n_features=52,
    suffix="_v2",
    v2_flag=True,
    v3_flag=False,
    prior_versions=[V1],
)

V3 = VersionConfig(
    version=3,
    label="V3 — posicional + tática + look-ahead (67 features)",
    n_features=67,
    suffix="_v3",
    v2_flag=True,
    v3_flag=True,
    prior_versions=[V1, V2],
)

V4 = VersionConfig(
    version=4,
    label="V4 — XGBoost + threshold tuning (67 features)",
    n_features=67,
    suffix="_v4",
    v2_flag=True,
    v3_flag=True,
    prior_versions=[V1, V2, V3],
    features_suffix="_v3",
)

ALL_VERSIONS = [V1, V2, V3, V4]
