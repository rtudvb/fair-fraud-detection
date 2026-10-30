"""Column schema of the BAF datasets and a lightweight validator.

The 30 features, ``month`` and the label follow the BAF datasheet (Jesus et al., 2022).
Variants III and V append two extra synthetic columns; they are allowed and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LABEL = "fraud_bool"
MONTH = "month"
PROTECTED = "customer_age"

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
)

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns where negative values encode "missing" according to the dataset card.
# EDA (notebooks/01_eda) checks this before any imputation decision is made.
NEGATIVE_MEANS_MISSING: tuple[str, ...] = (
    "prev_address_months_count",
    "current_address_months_count",
    "intended_balcon_amount",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)

N_MONTHS = 8


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_frame`."""

    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_frame(df: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Check that ``df`` looks like a BAF dataset.

    Errors: missing required columns, a non-binary label, months outside 0-7, missing values.
    Warnings: extra columns (expected for Variants III and V).
    With ``strict=True`` any error raises :class:`ValueError`.
    """
    report = ValidationReport(n_rows=len(df))
    required = (*FEATURES, MONTH, LABEL)

    missing = [col for col in required if col not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")

    extra = [col for col in df.columns if col not in required]
    if extra:
        report.warnings.append(f"extra columns (kept as features): {extra}")

    if LABEL in df.columns:
        labels = set(pd.unique(df[LABEL]))
        if not labels <= {0, 1}:
            report.errors.append(f"label {LABEL!r} must be 0/1, found {sorted(labels)}")

    if MONTH in df.columns:
        months = df[MONTH]
        if months.min() < 0 or months.max() >= N_MONTHS:
            report.errors.append(f"{MONTH!r} must lie in 0..{N_MONTHS - 1}")

    present = [col for col in required if col in df.columns]
    na_counts = df[present].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if not na_counts.empty:
        report.errors.append(f"missing values in: {na_counts.to_dict()}")

    if strict and not report.ok:
        raise ValueError("; ".join(report.errors))
    return report
