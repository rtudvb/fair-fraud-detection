"""Feature preprocessing shared by the model wrappers.

Every model receives the same pandas frame: numeric columns plus categorical columns in
``category`` dtype. Tree boosters and foundation models consume categories natively;
linear models and the MLP get one-hot encoding plus scaling. Column roles are detected
from dtypes at fit time, so the extra columns of BAF Variants III and V work unchanged.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

categorical_columns = make_column_selector(dtype_include="category")
numeric_columns = make_column_selector(dtype_exclude="category")


class CategoryAligner(BaseEstimator, TransformerMixin):
    """Fix the category levels seen in training so train and test share one encoding.

    Levels unseen during training become missing values, which every downstream model
    handles (boosters natively, encoders via ``handle_unknown``).
    """

    def fit(self, X: pd.DataFrame, y=None) -> CategoryAligner:
        self.categories_ = {
            col: X[col].astype("category").cat.categories
            for col in X.columns
            if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object
        }
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_names_in_ if col not in X.columns]
        if missing:
            raise KeyError(f"columns missing at transform time: {missing}")
        X = X[self.feature_names_in_].copy()
        for col, levels in self.categories_.items():
            values = X[col].astype(object)
            values = values.where(values.isin(levels))  # unseen levels -> missing
            X[col] = pd.Categorical(values, categories=levels)
        return X

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)


def one_hot_and_scale() -> ColumnTransformer:
    """One-hot categories, standardise numbers (logistic regression, MLP)."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("num", StandardScaler(), numeric_columns),
        ]
    )


def ordinal_encode() -> ColumnTransformer:
    """Integer-code categories, pass numbers through (random forest)."""
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                categorical_columns,
            ),
            ("num", "passthrough", numeric_columns),
        ]
    )
