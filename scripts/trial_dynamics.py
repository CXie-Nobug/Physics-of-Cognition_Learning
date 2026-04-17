import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def fit_trial_linear(trial_idx, vals):
    """
    Fit linear relation: vals ~ trial_idx

    Returns
    -------
    res : dict
        {
            "slope": ...,
            "intercept": ...,
            "rvalue": ...,
            "pvalue": ...,
            "stderr": ...,
            "pred": ...,
            "n": ...
        }
    """
    trial_idx = np.asarray(trial_idx, dtype=float)
    vals = np.asarray(vals, dtype=float)

    valid = np.isfinite(trial_idx) & np.isfinite(vals)
    x = trial_idx[valid]
    y = vals[valid]

    if len(x) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "rvalue": np.nan,
            "pvalue": np.nan,
            "stderr": np.nan,
            "pred": np.full_like(trial_idx, np.nan, dtype=float),
            "n": len(x),
        }

    fit = linregress(x, y)
    pred = fit.intercept + fit.slope * trial_idx

    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "rvalue": fit.rvalue,
        "pvalue": fit.pvalue,
        "stderr": fit.stderr,
        "pred": pred,
        "n": len(x),
    }

def build_component_colors(component_windows):
    comps = list(component_windows.keys())

    # -----------------------------
    # 1) Group by P and N
    # -----------------------------
    def parse_comp(comp):
        match = re.match(r"([PN])(\d+)", comp)
        if match:
            typ = match.group(1)
            num = int(match.group(2))
        else:
            typ = "OTHER"
            num = 0
        return typ, num

    parsed = [(comp, *parse_comp(comp)) for comp in comps]

    P_comps = sorted([x for x in parsed if x[1] == "P"], key=lambda x: x[2])
    N_comps = sorted([x for x in parsed if x[1] == "N"], key=lambda x: x[2])

    # -----------------------------
    # 2) generate colors for each group
    # -----------------------------
    def generate_colors(n, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        return [mcolors.to_hex(cmap(0.4 + 0.5 * i / max(n-1, 1))) for i in range(n)]

    P_colors = generate_colors(len(P_comps), "Oranges")
    N_colors = generate_colors(len(N_comps), "Blues")

    # -----------------------------
    # 3)  dict
    # -----------------------------
    comp_colors = {}

    for (comp, _, _), color in zip(P_comps, P_colors):
        comp_colors[comp] = color

    for (comp, _, _), color in zip(N_comps, N_colors):
        comp_colors[comp] = color

    # fallback
    for comp in comps:
        if comp not in comp_colors:
            comp_colors[comp] = "gray"

    return comp_colors

def plot_group_trial_dynamics(
    erp_roi_group_roll_df,
    component_windows,
    CONDITION=None,
    group_col="region",
    b_plot_channel_seperate=True,
    feature_specs=None,
    comp_colors=None,
    linestyle_map=None,
    figsize_per_row=(8, 3),
    b_fit_linear=True,
    b_plot_fit=False,
):
    """
    Plot grand-average rolling ERP features.

    Returns
    -------
    res_df : pd.DataFrame
        Linear fitting results for each group × component × metric
    """

    if feature_specs is None:
        feature_specs = [
            ("mean_amp_roll_mean",   "mean_amp_roll_sem",   "Mean amplitude (a.u.)", "Mean amplitude"),
            ("peak_amp_roll_mean",   "peak_amp_roll_sem",   "Peak amplitude (a.u.)", "Peak amplitude"),
            ("peak_lat_ms_roll_mean","peak_lat_ms_roll_sem","Latency (ms)",          "Peak latency"),
        ]

    if comp_colors is None:
        comp_colors = build_component_colors(component_windows)

    if group_col not in erp_roi_group_roll_df.columns:
        raise ValueError(f"`{group_col}` not found in dataframe columns")

    group_names = list(erp_roi_group_roll_df[group_col].dropna().unique())
    if len(group_names) == 0:
        raise ValueError(f"No valid groups found in column `{group_col}`")

    if linestyle_map is None:
        base_styles = ["-", "--", ":", "-."]
        linestyle_map = {
            g: base_styles[i % len(base_styles)]
            for i, g in enumerate(group_names)
        }

    # ==========================================
    # collect fitting results
    # ==========================================
    fit_rows = []

    # ======================================================
    # Mode 1: separate subplot for each group
    # ======================================================
    if b_plot_channel_seperate:
        n_rows = len(feature_specs)
        n_cols = len(group_names)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_row[0] * n_cols, figsize_per_row[1] * n_rows),
            sharex=True,
            squeeze=False
        )

        for c, group_name in enumerate(group_names):
            df_g = (
                erp_roi_group_roll_df[erp_roi_group_roll_df[group_col] == group_name]
                .sort_values("trial_index")
                .reset_index(drop=True)
            )

            trial_idx = df_g["trial_index"].values

            for r, (mean_suffix, sem_suffix, ylab, title_feat) in enumerate(feature_specs):
                ax = axes[r, c]

                for comp in component_windows.keys():
                    mean_col = f"{comp}_{mean_suffix}"
                    sem_col  = f"{comp}_{sem_suffix}"

                    if mean_col not in df_g.columns or sem_col not in df_g.columns:
                        print(f"Skip {group_name}-{comp}: missing {mean_col} or {sem_col}")
                        continue

                    mean_vals = df_g[mean_col].values
                    sem_vals  = df_g[sem_col].values
                    color = comp_colors.get(comp, None)

                    ax.plot(
                        trial_idx,
                        mean_vals,
                        linewidth=2.0,
                        color=color,
                        linestyle="-",
                        label=comp
                    )

                    ax.fill_between(
                        trial_idx,
                        mean_vals - sem_vals,
                        mean_vals + sem_vals,
                        alpha=0.18,
                        color=color,
                        linewidth=0
                    )

                    # ------------------------------
                    # linear fit
                    # ------------------------------
                    if b_fit_linear:
                        fit_res = fit_trial_linear(trial_idx, mean_vals)

                        fit_rows.append({
                            group_col: group_name,
                            "component": comp,
                            "metric_suffix": mean_suffix,
                            "metric_title": title_feat,
                            "mean_col": mean_col,
                            "slope": fit_res["slope"],
                            "intercept": fit_res["intercept"],
                            "rvalue": fit_res["rvalue"],
                            "r2": fit_res["rvalue"]**2 if np.isfinite(fit_res["rvalue"]) else np.nan,
                            "pvalue": fit_res["pvalue"],
                            "stderr": fit_res["stderr"],
                            "n": fit_res["n"],
                        })

                        if b_plot_fit:
                            ax.plot(
                                trial_idx,
                                fit_res["pred"],
                                color=color,
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.9,
                            )

                if "latency" not in title_feat.lower():
                    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)

                if c == 0:
                    ax.set_ylabel(ylab, fontsize=10)

                ax.set_title(f"{group_name} — {title_feat}", fontsize=11)
                ax.grid(True, alpha=0.25)

                if r == len(feature_specs) - 1:
                    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0), fontsize=9)

        for ax in axes[-1, :]:
            ax.set_xlabel("Trial index", fontsize=11)

        title_prefix = CONDITION if CONDITION is not None else "ERP"
        fig.suptitle(
            f"{title_prefix} — Group Trial-by-Trial Neural Dynamics\n"
            f"(separate {group_col} panels; subject-wise rolling mean, then averaged across subjects)",
            fontsize=13
        )
        fig.tight_layout()
        plt.show()

    # ======================================================
    # Mode 2: overlay all groups in one subplot per feature
    # ======================================================
    else:
        n_plots = len(feature_specs)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(8, 3 * n_plots),
            sharex=True
        )

        if n_plots == 1:
            axes = [axes]

        for ax, (mean_suffix, sem_suffix, ylab, title_feat) in zip(axes, feature_specs):
            for group_name in group_names:
                df_g = (
                    erp_roi_group_roll_df[erp_roi_group_roll_df[group_col] == group_name]
                    .sort_values("trial_index")
                    .reset_index(drop=True)
                )

                trial_idx = df_g["trial_index"].values
                ls = linestyle_map[group_name]

                for comp in component_windows.keys():
                    mean_col = f"{comp}_{mean_suffix}"
                    sem_col  = f"{comp}_{sem_suffix}"

                    if mean_col not in df_g.columns or sem_col not in df_g.columns:
                        print(f"Skip {group_name}-{comp}: missing {mean_col} or {sem_col}")
                        continue

                    mean_vals = df_g[mean_col].values
                    sem_vals  = df_g[sem_col].values
                    color = comp_colors.get(comp, None)

                    ax.plot(
                        trial_idx,
                        mean_vals,
                        linewidth=2.0,
                        color=color,
                        linestyle=ls,
                        label=f"{comp}-{group_name}"
                    )

                    ax.fill_between(
                        trial_idx,
                        mean_vals - sem_vals,
                        mean_vals + sem_vals,
                        alpha=0.08,
                        color=color,
                        linewidth=0
                    )

                    # ------------------------------
                    # linear fit
                    # ------------------------------
                    if b_fit_linear:
                        fit_res = fit_trial_linear(trial_idx, mean_vals)

                        fit_rows.append({
                            group_col: group_name,
                            "component": comp,
                            "metric_suffix": mean_suffix,
                            "metric_title": title_feat,
                            "mean_col": mean_col,
                            "slope": fit_res["slope"],
                            "intercept": fit_res["intercept"],
                            "rvalue": fit_res["rvalue"],
                            "r2": fit_res["rvalue"]**2 if np.isfinite(fit_res["rvalue"]) else np.nan,
                            "pvalue": fit_res["pvalue"],
                            "stderr": fit_res["stderr"],
                            "n": fit_res["n"],
                        })

                        if b_plot_fit:
                            ax.plot(
                                trial_idx,
                                fit_res["pred"],
                                color=color,
                                linestyle=":",
                                linewidth=1.2,
                                alpha=0.9,
                            )

            if "latency" not in title_feat.lower():
                ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)

            ax.set_ylabel(ylab, fontsize=10)
            ax.set_title(f"Group trend — {title_feat}", fontsize=11)
            ax.grid(True, alpha=0.25)
            if ax == axes[-1]:
                ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0), fontsize=8)

        axes[-1].set_xlabel("Trial index", fontsize=11)

        title_prefix = CONDITION if CONDITION is not None else "ERP"
        fig.suptitle(
            f"{title_prefix} — Group Trial-by-Trial Neural Dynamics\n"
            f"(overlay {group_col}; color=component, linestyle={group_col})",
            fontsize=13
        )
        fig.tight_layout()
        plt.show()

    res_df = pd.DataFrame(fit_rows)
    return res_df