import logging
import pandas as pd

logger = logging.getLogger(__name__)


def generate_tropical_cyclone_future_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate future extreme windspeed values using simple site-based scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame expected to contain baseline extreme windspeed columns and
        optionally ``Elevation (meter above sea level)``.

    Returns
    -------
    pandas.DataFrame
        Input dataframe with additional columns for base and worst case
        extreme windspeed scenarios.
    """
    required_cols = [
        "Extreme Windspeed 10 year Return Period (km/h)",
        "Extreme Windspeed 20 year Return Period (km/h)",
        "Extreme Windspeed 50 year Return Period (km/h)",
        "Extreme Windspeed 100 year Return Period (km/h)",
    ]

    if not any(col in df.columns for col in required_cols):
        logger.warning("No baseline tropical cyclone columns found. Skipping future analysis.")
        return df

    # Default scaling factors if elevation is not available
    base_factor = 1.05
    worst_factor = 1.10

    if "Elevation (meter above sea level)" in df.columns:
        elev = pd.to_numeric(
            df["Elevation (meter above sea level)"], errors="coerce"
        ).fillna(0)
        base_factor = 1 + (elev / 1000) * 0.05
        worst_factor = 1 + (elev / 1000) * 0.10

    for col in required_cols:
        base_col = f"{col} - Base Case"
        worst_col = f"{col} - Worst Case"

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        df[base_col] = (numeric_col * base_factor).round(1)
        df[worst_col] = (numeric_col * worst_factor).round(1)

    return df