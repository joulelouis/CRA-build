import os
import math
import pandas as pd
from django.conf import settings


def generate_future_water_stress_from_baseline(baseline_csv_path: str) -> dict:
    """Add 2030 and 2050 water stress projections using ``pfaf_id``.

    Parameters
    ----------
    baseline_csv_path : str
        Path to the facility CSV that must contain a ``pfaf_id`` column.

    Returns
    -------
    dict
        Dictionary with ``output_csv`` key pointing to the written file,
        or ``error`` key if an exception occurred.
    """
    try:
        water_dir = os.path.join(
            settings.BASE_DIR, "water_stress", "static", "input_files"
        )
        output_dir = os.path.join(
            settings.BASE_DIR, "climate_hazards_analysis", "static", "input_files"
        )
        os.makedirs(output_dir, exist_ok=True)

        future_csv_path = os.path.join(
            water_dir, "Aqueduct40_future_annual_y2023m07d05.csv"
        )

        df_baseline = pd.read_csv(baseline_csv_path)
        df_future = pd.read_csv(future_csv_path)

        # Normalize ``pfaf_id`` types to ensure successful merge
        df_baseline["pfaf_id"] = pd.to_numeric(df_baseline["pfaf_id"], errors="coerce").astype("Int64")

        if "pfaf_id" not in df_baseline.columns:
            raise ValueError("Baseline CSV must contain 'pfaf_id' column.")

        df_future_selected = df_future[
            [
                "pfaf_id",
                "bau30_ws_x_r",
                "bau50_ws_x_r",
                "pes30_ws_x_r",
                "pes50_ws_x_r",
            ]
        ].copy()
        
        df_future_selected["pfaf_id"] = pd.to_numeric(
            df_future_selected["pfaf_id"], errors="coerce"
        ).astype("Int64")
        # Convert future water stress ratios to percentages keeping one decimal
        # place. ``mul(100)`` scales values while ``round(1)`` preserves the
        # decimal precision used in the baseline dataset.

        df_future_selected["bau30_ws_x_r"] = (
            df_future_selected["bau30_ws_x_r"].fillna(0).mul(100).round(1)
        )
        df_future_selected["bau50_ws_x_r"] = (
            df_future_selected["bau50_ws_x_r"].fillna(0).mul(100).round(1)
        )
        df_future_selected["pes30_ws_x_r"] = (
            df_future_selected["pes30_ws_x_r"].fillna(0).mul(100).round(1)
        )
        df_future_selected["pes50_ws_x_r"] = (
            df_future_selected["pes50_ws_x_r"].fillna(0).mul(100).round(1)
        )

        df_merged = pd.merge(df_baseline, df_future_selected, on="pfaf_id", how="left")

        df_merged.rename(
            columns={
                "bau30_ws_x_r": "Water Stress Exposure 2030 (%) - Moderate Case",
                "bau50_ws_x_r": "Water Stress Exposure 2050 (%) - Moderate Case",
                "pes30_ws_x_r": "Water Stress Exposure 2030 (%) - Worst Case",
                "pes50_ws_x_r": "Water Stress Exposure 2050 (%) - Worst Case",
            },
            inplace=True,
        )

        output_csv = os.path.join(output_dir, "future_water_stress_output.csv")
        df_merged.to_csv(output_csv, index=False)

        print(f"✅ Future water stress output saved: {output_csv}")
        return {"output_csv": output_csv}

    except Exception as e:  # pragma: no cover - small helper function
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def apply_future_water_stress_to_csv(
    input_csv: str, output_csv: str | None = None
) -> str:
    """Apply :func:`generate_future_water_stress_from_baseline` to a CSV file."""
    res = generate_future_water_stress_from_baseline(input_csv)
    out_path = res.get("output_csv")
    if out_path is None:
        raise RuntimeError(res.get("error", "Future water stress generation failed"))
    if output_csv is not None and out_path != output_csv:
        df = pd.read_csv(out_path)
        df.to_csv(output_csv, index=False)
        return output_csv
    return out_path