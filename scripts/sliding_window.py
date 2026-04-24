import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts import global_settings

def aggregate(
    erp_features_all_df: pd.DataFrame,
    component_windows: dict,
    CONDITION: str,
    ROI_NAME: str=None,
    meta_cols=None,
):
    # =========================================================
    # 0) Check CONDITION
    # =========================================================
    regions_of_interest = global_settings.roi_region_dict.get(ROI_NAME, global_settings.roi_region_dict["all_rois_flat"])
    if len(regions_of_interest) == 0:
        raise ValueError(f"No regions found for CONDITION={ROI_NAME}")

    print(f"CONDITION = {ROI_NAME}")
    print("Subregions:", list(regions_of_interest.keys()))

    # =========================================================
    # 1) 自动识别 feature columns（排除 meta cols）
    # =========================================================
    if meta_cols is None:
        meta_cols = [
            "sub_id",
            "trial_index",
            "channel",
            "condition",
            "region",
        ]

    missing_meta_cols = [col for col in ["sub_id", "trial_index", "channel"] if col not in erp_features_all_df.columns]
    if len(missing_meta_cols) > 0:
        raise ValueError(
            f"erp_features_all_df is missing required meta columns: {missing_meta_cols}"
        )

    feature_cols = [
        col for col in erp_features_all_df.columns
        if col not in meta_cols
    ]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after excluding meta_cols.")

    # 只保留数值列，避免 object / string 列被错误聚合
    non_numeric_feature_cols = [
        col for col in feature_cols
        if not pd.api.types.is_numeric_dtype(erp_features_all_df[col])
    ]
    if len(non_numeric_feature_cols) > 0:
        print(
            f"Warning: these non-numeric columns are excluded from feature_cols: "
            f"{non_numeric_feature_cols}"
        )

    feature_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(erp_features_all_df[col])
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found after excluding meta_cols.")

    print(f"Detected {len(feature_cols)} feature columns.")
    print("Feature columns:", feature_cols)

    # =========================================================
    # 2) 按 subregion 分开计算
    # =========================================================
    all_region_group_roll_df = []
    all_region_subject_roll_df = []
    all_region_subject_feature_df = []

    for region_name, roi_channels in regions_of_interest.items():
        print(f"\nProcessing region = {region_name}, channels = {roi_channels}")

        # -----------------------------------------------------
        # 2.1) 先筛这个 subregion 的 channel
        # -----------------------------------------------------
        df_roi = erp_features_all_df[
            erp_features_all_df["channel"].isin(roi_channels)
        ].copy()

        if len(df_roi) == 0:
            print(f"  Skipped: no data found for region {region_name}")
            continue

        # -----------------------------------------------------
        # 2.2) 按 sub_id + trial_index 聚合 ROI 内多个电极
        # -----------------------------------------------------
        agg_dict = {col: "mean" for col in feature_cols}

        erp_region_features_df = (
            df_roi
            .groupby(["sub_id", "trial_index"], observed=True, as_index=False)
            .agg(agg_dict)
            .sort_values(["sub_id", "trial_index"])
            .reset_index(drop=True)
        )

        erp_region_features_df["condition"] = CONDITION
        erp_region_features_df["region"] = region_name

        all_region_subject_feature_df.append(erp_region_features_df)

        # -----------------------------------------------------
        # 2.3) 每个被试内部按 trial 做 rolling
        # -----------------------------------------------------
        region_roll_list = []

        for sub_id in erp_region_features_df["sub_id"].unique():
            df_sub = erp_region_features_df.query("sub_id == @sub_id").copy()
            if len(df_sub) == 0:
                continue

            df_sub = df_sub.sort_values("trial_index").reset_index(drop=True)

            for col in feature_cols:
                roll_col = f"{col}_roll"
                df_sub[roll_col] = df_sub[col].rolling(
                    window=global_settings.ROLLING_WINDOW,
                    center=True,
                    min_periods=1
                ).mean()

            df_sub["condition"] = CONDITION
            df_sub["region"] = region_name

            region_roll_list.append(df_sub)

        if len(region_roll_list) == 0:
            print(f"  Skipped: no rolling result for region {region_name}")
            continue

        erp_region_roll_df = pd.concat(region_roll_list, ignore_index=True)
        all_region_subject_roll_df.append(erp_region_roll_df)

        # -----------------------------------------------------
        # 2.4) across-subject grand average
        # -----------------------------------------------------
        group_rows = []

        for trial_index, df_trial in erp_region_roll_df.groupby("trial_index", observed=True):
            row = {
                "trial_index": trial_index,
                "condition": CONDITION,
                "region": region_name,
            }

            for col in feature_cols:
                roll_col = f"{col}_roll"
                row[f"{roll_col}_mean"] = df_trial[roll_col].mean()
                row[f"{roll_col}_sem"] = df_trial[roll_col].sem()

            group_rows.append(row)

        erp_region_group_roll_df = (
            pd.DataFrame(group_rows)
            .sort_values("trial_index")
            .reset_index(drop=True)
        )

        all_region_group_roll_df.append(erp_region_group_roll_df)

    # =========================================================
    # 3) 合并所有 subregion 的结果
    # =========================================================
    if len(all_region_subject_feature_df) > 0:
        erp_roi_features_df = pd.concat(all_region_subject_feature_df, ignore_index=True)
    else:
        erp_roi_features_df = pd.DataFrame()

    if len(all_region_subject_roll_df) > 0:
        erp_roi_roll_df = pd.concat(all_region_subject_roll_df, ignore_index=True)
    else:
        erp_roi_roll_df = pd.DataFrame()

    if len(all_region_group_roll_df) > 0:
        erp_roi_group_roll_df = pd.concat(all_region_group_roll_df, ignore_index=True)
        erp_roi_group_roll_df = erp_roi_group_roll_df.sort_values(
            ["region", "trial_index"]
        ).reset_index(drop=True)
    else:
        erp_roi_group_roll_df = pd.DataFrame()

    # =========================================================
    # 4) display
    # =========================================================
    print("\n[Per-subject region-aggregated features]")
    display(erp_roi_features_df.head())

    print("\n[Per-subject rolling features]")
    display(erp_roi_roll_df.head())

    print("\n[Grand average rolling features]")
    display(erp_roi_group_roll_df.head())

    return erp_roi_features_df, erp_roi_roll_df, erp_roi_group_roll_df