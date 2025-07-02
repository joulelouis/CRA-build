import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_tropical_cyclone_future_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate future extreme windspeed values for 2050 scenarios.

    The calculation applies constant percentage increases for SSP245 and
    SSP585 ("Base" and "Worst" cases) and scales the values based on site
    elevation when available.
    """
    pattern = re.compile(r"Extreme\s+Windspeed.*return period.*\(km/h\)", re.IGNORECASE)
    ew_cols = [col for col in df.columns if pattern.search(col)]

    if len(ew_cols) != 4:
        logger.warning(
            "Found %d return-period columns; expected 4. Matches: %s", len(ew_cols), ew_cols
        )
        return df

    def extract_year(col: str) -> int:
        m = re.search(r"(\d+)", col)
        return int(m.group(1)) if m else 0

    ew_cols.sort(key=extract_year)
    df[ew_cols] = df[ew_cols].apply(pd.to_numeric, errors="coerce")

    pct_base = 0.0351  # RCP4.5 @2050
    pct_worst = 0.0493  # RCP8.5 @2050

    base_factor = 1.05
    worst_factor = 1.10

    if "Elevation (meter above sea level)" in df.columns:
        elev = pd.to_numeric(df["Elevation (meter above sea level)"], errors="coerce").fillna(0)
        base_factor = 1 + (elev / 1000) * 0.05
        worst_factor = 1 + (elev / 1000) * 0.10

    for col in ew_cols:
        base_col = f"{col} - Base Case"
        worst_col = f"{col} - Worst Case"

        numeric_col = df[col]
        df[base_col] = (
            np.ceil(numeric_col * (1 + pct_base) * base_factor)
            .astype("Int64")
        )
        df[worst_col] = (
            np.ceil(numeric_col * (1 + pct_worst) * worst_factor)
            .astype("Int64")
        )

    future_cols = [f"{c} - Base Case" for c in ew_cols] + [f"{c} - Worst Case" for c in ew_cols]

    if "Extreme Windspeed 100 year Return Period (km/h)" in df.columns:
        cols = [c for c in df.columns if c not in future_cols]
        insert_pos = cols.index("Extreme Windspeed 100 year Return Period (km/h)") + 1
        for col in future_cols:
            cols.insert(insert_pos, col)
            insert_pos += 1
        df = df[cols]

    return df

def apply_future_windspeeds_to_csv(input_csv: str, output_csv: str | None = None) -> str:
    """Apply future windspeed calculations to an existing CSV.

    Parameters
    ----------
    input_csv : str
        Path to ``combined_output.csv`` or similar file.
    output_csv : str, optional
        Where to write the file with additional future windspeed columns. If not
        provided, ``_v2`` is appended to the input filename.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    in_path = Path(input_csv)
    if output_csv is None:
        output_csv = str(in_path.with_name(f"{in_path.stem}_v2.csv"))

    df = pd.read_csv(in_path)
    df = generate_tropical_cyclone_future_analysis(df)
    df.to_csv(output_csv, index=False)
    return output_csv