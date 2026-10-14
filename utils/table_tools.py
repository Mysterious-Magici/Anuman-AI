"""Safe, fixed table operations (replaces the old LLM-generated-code 'Action Agent')."""
from __future__ import annotations

import json
import operator as _op

import pandas as pd

OPERATIONS = {
    "drop_missing": "Drop rows with missing values",
    "drop_duplicates": "Drop duplicate rows",
    "fill_median": "Fill missing numeric values with the column median",
    "filter": "Keep rows matching a condition",
}
COMPARISONS = ["==", "!=", ">", ">=", "<", "<=", "contains"]
_FUNCS = {"==": _op.eq, "!=": _op.ne, ">": _op.gt, ">=": _op.ge, "<": _op.lt, "<=": _op.le}


def apply_operation(df: pd.DataFrame, op: str, column=None, comparison=None, value=None) -> pd.DataFrame:
    """Return a NEW DataFrame; the original is never modified."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("A valid DataFrame is required.")
    if op == "drop_missing":
        return df.dropna()
    if op == "drop_duplicates":
        try:
            return df.drop_duplicates()
        except TypeError:
            def _to_hashable(v):
                return json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
            temp = df.map(_to_hashable)
            return df.loc[temp.drop_duplicates().index].copy()
    if op == "fill_median":
        out = df.copy()
        num = out.select_dtypes("number").columns
        if not num.empty:
            out[num] = out[num].fillna(out[num].median())
        return out
    if op == "filter":
        if column not in df.columns:
            raise ValueError("Choose a column to filter on.")
        if comparison not in COMPARISONS:
            raise ValueError("Choose a valid condition.")
        series = df[column]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if comparison == "contains":
            mask = series.astype(str).str.contains(str(value or ""), case=False, na=False, regex=False)
        elif pd.api.types.is_numeric_dtype(series):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{value}' is not a number, but column '{column}' is numeric.") from None
            mask = _FUNCS[comparison](series, number)
        else:
            mask = _FUNCS[comparison](series.astype(str), str(value if value is not None else ""))
        return df[mask]
    raise ValueError("Unknown operation.")