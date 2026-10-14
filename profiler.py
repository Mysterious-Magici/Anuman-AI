import json
import pandas as pd


class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def generate_profile(self) -> dict:
        """Runs rule-based checks and returns a summary dictionary."""
        # Safeguard duplicate rows calculation against unhashable types (e.g. lists/dicts in cells)
        dup_count = 0
        try:
            dup_count = int(self.df.duplicated().sum())
        except TypeError:
            try:
                def _safe_str(v):
                    return json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v)
                dup_count = int(self.df.map(_safe_str).duplicated().sum())
            except Exception:
                dup_count = 0

        profile = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": [str(c) for c in self.df.columns],
            "duplicate_rows": dup_count,
            "missing_values": {},
            "data_types": {}
        }

        # Use positional indexing to safely handle duplicate column names
        seen_names = {}
        for i in range(len(self.df.columns)):
            raw_name = str(self.df.columns[i])
            if raw_name in seen_names:
                seen_names[raw_name] += 1
                col_key = f"{raw_name}_{seen_names[raw_name]}"
            else:
                seen_names[raw_name] = 1
                col_key = raw_name

            series = self.df.iloc[:, i]
            missing = int(series.isnull().sum())
            if missing > 0:
                profile["missing_values"][col_key] = missing

            profile["data_types"][col_key] = str(series.dtype)

        return profile

    def get_sample_data(self, n=3):
        """Returns a small sample to give the LLM context of the actual data."""
        if self.df.empty:
            return []
        n = max(1, n)
        return self.df.head(n).to_dict(orient='records')