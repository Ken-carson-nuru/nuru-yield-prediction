import pandas as pd


def normalize_raw_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    cols = {c.strip(): c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Plot identifier
    if "plot_id" not in df.columns:
        if "Plot_No" in df.columns:
            df = df.rename(columns={"Plot_No": "plot_id"})
        elif "plot_no" in df.columns:
            df = df.rename(columns={"plot_no": "plot_id"})

    # Planting date
    if "Planting_Date" not in df.columns:
        if "planting_date" in df.columns:
            df = df.rename(columns={"planting_date": "Planting_Date"})
        elif "boxes_planting_date" in df.columns:
            df = df.rename(columns={"boxes_planting_date": "Planting_Date"})

    # Crop type
    if "crop_type" not in df.columns and "boxes_crop" in df.columns:
        df = df.rename(columns={"boxes_crop": "crop_type"})

    # Numeric measurements
    for c in [
        "wet_kgs_crop_box1",
        "wet_kgs_crop_box2",
        "dry_box1_dry_weight",
        "dry_box2_dry_weight",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dates
    if "Planting_Date" in df.columns:
        df["Planting_Date"] = pd.to_datetime(df["Planting_Date"], errors="coerce")
    if "wet_completed_time" in df.columns:
        df["wet_completed_time"] = pd.to_datetime(df["wet_completed_time"], errors="coerce")

    return df


def compute_labels_from_raw(df: pd.DataFrame, box_area_m2: float = 40.0, boxes_count: int = 2) -> pd.DataFrame:
    df = normalize_raw_labels(df)

    total_area = box_area_m2 * boxes_count
    m2_per_hectare = 10000.0

    # Compute kg/ha if source columns exist
    wet_cols_present = {c for c in ["wet_kgs_crop_box1", "wet_kgs_crop_box2"] if c in df.columns}
    dry_cols_present = {c for c in ["dry_box1_dry_weight", "dry_box2_dry_weight"] if c in df.columns}

    if wet_cols_present:
        w1 = df["wet_kgs_crop_box1"] if "wet_kgs_crop_box1" in df.columns else 0
        w2 = df["wet_kgs_crop_box2"] if "wet_kgs_crop_box2" in df.columns else 0
        df["wet_harvest_kg/ha"] = ((w1 + w2) / total_area) * m2_per_hectare

    if dry_cols_present:
        d1 = df["dry_box1_dry_weight"] if "dry_box1_dry_weight" in df.columns else 0
        d2 = df["dry_box2_dry_weight"] if "dry_box2_dry_weight" in df.columns else 0
        df["dry_harvest_kg/ha"] = ((d1 + d2) / total_area) * m2_per_hectare

    # Round for consistency
    for t in ["wet_harvest_kg/ha", "dry_harvest_kg/ha"]:
        if t in df.columns:
            df[t] = df[t].round(2)

    # Keep key columns for downstream merge
    keep = [
        "plot_id",
        "Planting_Date",
        "crop_type",
        "wet_harvest_kg/ha",
        "dry_harvest_kg/ha",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()

