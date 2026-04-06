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

    @property
    def features_csv(self) -> Path:
        return DATA_DIR / "features" / f"features{self.suffix}.csv"

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
        """Return (decision_tree, random_forest, feature_names)."""
        with open(self.models_dir / "feature_names.json") as f:
            feature_names = json.load(f)
        dt = joblib.load(self.models_dir / "decision_tree.joblib")
        rf = joblib.load(self.models_dir / "random_forest.joblib")
        return dt, rf, feature_names

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

ALL_VERSIONS = [V1, V2, V3]
