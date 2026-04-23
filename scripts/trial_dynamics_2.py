import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations


# =========================================================
# 0) small helpers
# =========================================================
def _safe_zscore(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x, ddof=0)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / s


def _fit_linear_residual(x, y):
    """
    Residualize y ~ 1 + x using simple linear regression.
    Returns residuals, beta
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    res = np.full_like(y, np.nan, dtype=float)

    if mask.sum() < 3:
        return res, {"intercept": np.nan, "slope": np.nan}

    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    y_hat = X @ beta
    res[mask] = y[mask] - y_hat

    return res, {"intercept": beta[0], "slope": beta[1]}


def _fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def _fisher_z_inv(z):
    return np.tanh(z)


def _corr_matrix(X):
    """
    X: (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    return np.corrcoef(X, rowvar=False)


def _make_named_feature_blocks(feature_cols):
    """
    Split features into named blocks by column name.

    Rules
    -----
    amp  : contains 'amp'
    lat  : contains 'lat'
    freq : contains 'power', 'freq', 'band'
    """
    amp_cols = [c for c in feature_cols if "amp" in c.lower()]
    lat_cols = [c for c in feature_cols if "lat" in c.lower()]
    freq_cols = [
        c for c in feature_cols
        if (
            ("power" in c.lower())
            or ("freq" in c.lower())
            or ("band" in c.lower())
        )
    ]

    block_dict = {}
    if len(amp_cols) > 0:
        block_dict["amp"] = amp_cols
    if len(lat_cols) > 0:
        block_dict["lat"] = lat_cols
    if len(freq_cols) > 0:
        block_dict["freq"] = freq_cols

    return block_dict

def _component_from_name(col):
    """
    Try extracting component label from feature name.
    Example:
        P50_mean_amp -> P50
        N100_peak_lat_ms -> N100
    """
    m = re.match(r"([A-Za-z]+\d+)", col)
    if m:
        return m.group(1)
    return col


# =========================================================
# 1) within-subject residualization
# =========================================================
def residualize_features_within_subject(
    df,
    feature_cols,
    region_col="region",
    sub_col="sub_id",
    trial_col="trial_index",
    suffix="_resid",
    zscore_within_subject=False,
):
    """
    For each subject x region x feature:
        feature ~ trial_index
    and return residualized dataframe.

    Parameters
    ----------
    zscore_within_subject : bool
        If True, z-score each feature within subject-region BEFORE residualization.
        Useful when subjects have very different scales.

    Returns
    -------
    df_out : DataFrame
        Original df plus residualized columns.
    beta_df : DataFrame
        Slope/intercept summary for each subject-region-feature.
    """
    df_out = df.copy()
    beta_rows = []

    for feat in feature_cols:
        resid_col = f"{feat}{suffix}"
        df_out[resid_col] = np.nan

    group_keys = [sub_col, region_col]

    for (sub, region), g in df_out.groupby(group_keys, sort=False):
        g = g.sort_values(trial_col)
        x = g[trial_col].values

        for feat in feature_cols:
            y = g[feat].values.astype(float)

            if zscore_within_subject:
                y = _safe_zscore(y)

            resid, beta = _fit_linear_residual(x, y)

            df_out.loc[g.index, f"{feat}{suffix}"] = resid
            beta_rows.append({
                sub_col: sub,
                region_col: region,
                "feature": feat,
                "intercept": beta["intercept"],
                "slope": beta["slope"],
            })

    beta_df = pd.DataFrame(beta_rows)
    return df_out, beta_df


# =========================================================
# 2) group-level correlation matrix
# =========================================================
def compute_group_level_correlation(
    df,
    feature_cols,
    region_col="region",
    sub_col="sub_id",
    method="pearson",
    min_n=5,
    fisher_average=True,
):
    """
    Compute subject-level correlation matrix within each region,
    then aggregate to group-level.

    Returns
    -------
    results : dict
        results[region] = {
            "subject_corrs": list of corr matrices,
            "group_corr": averaged matrix,
            "group_z": averaged fisher-z matrix (if fisher_average),
            "n_subjects": int,
            "feature_cols": feature_cols,
        }
    """
    if method != "pearson":
        raise NotImplementedError("Currently only pearson is implemented.")

    results = {}

    for region, df_r in df.groupby(region_col, sort=False):
        subj_corrs = []

        for sub, g in df_r.groupby(sub_col, sort=False):
            X = g[feature_cols].apply(pd.to_numeric, errors="coerce").values
            valid_rows = np.isfinite(X).sum(axis=1) == X.shape[1]
            X = X[valid_rows]

            if X.shape[0] < max(min_n, 3):
                continue

            C = _corr_matrix(X)
            subj_corrs.append(C)

        if len(subj_corrs) == 0:
            results[region] = {
                "subject_corrs": [],
                "group_corr": None,
                "group_z": None,
                "n_subjects": 0,
                "feature_cols": feature_cols,
            }
            continue

        subj_corrs = np.stack(subj_corrs, axis=0)

        if fisher_average:
            z_mats = _fisher_z(subj_corrs)
            group_z = np.nanmean(z_mats, axis=0)
            group_corr = _fisher_z_inv(group_z)
        else:
            group_z = None
            group_corr = np.nanmean(subj_corrs, axis=0)

        results[region] = {
            "subject_corrs": subj_corrs,
            "group_corr": group_corr,
            "group_z": group_z,
            "n_subjects": subj_corrs.shape[0],
            "feature_cols": feature_cols,
        }

    return results


def test_group_level_correlations(
    corr_results,
    alpha=0.05,
):
    """
    One-sample t-test on Fisher-z across subjects for each feature pair.
    Useful if you want inferential stats on subject-level correlations.

    Returns
    -------
    stats_dict[region] = DataFrame with columns:
        feature_i, feature_j, mean_r, mean_z, t, p
    """
    stats_dict = {}

    for region, res in corr_results.items():
        subj_corrs = res["subject_corrs"]
        feats = res["feature_cols"]

        if len(subj_corrs) == 0:
            stats_dict[region] = pd.DataFrame()
            continue

        z_mats = _fisher_z(subj_corrs)
        rows = []

        n_feat = len(feats)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                zvals = z_mats[:, i, j]
                t, p = ttest_1samp(zvals, popmean=0.0, nan_policy="omit")
                rows.append({
                    "feature_i": feats[i],
                    "feature_j": feats[j],
                    "mean_z": np.nanmean(zvals),
                    "mean_r": _fisher_z_inv(np.nanmean(zvals)),
                    "t": t,
                    "p": p,
                })

        stats_dict[region] = pd.DataFrame(rows).sort_values("p")
    return stats_dict


def plot_group_correlation_matrices(corr_results, figsize_per_region=(6, 5), vmin=-1, vmax=1):
    regions = list(corr_results.keys())
    n = len(regions)
    if n == 0:
        return None, None

    fig, axes = plt.subplots(1, n, figsize=(figsize_per_region[0] * n, figsize_per_region[1]), squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    for ax, region in zip(axes, regions):
        res = corr_results[region]
        C = res["group_corr"]
        feats = res["feature_cols"]

        if C is None:
            ax.set_title(f"{region} (no data)")
            ax.axis("off")
            continue

        im = ax.imshow(C, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{region} (n={res['n_subjects']})")
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=45, fontsize=8, ha='right')
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, rotation=45, fontsize=8)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    
    return fig, axes


# =========================================================
# 3) PCA loadings + score-vs-trial
# =========================================================
def run_group_pca_by_region(
    df,
    feature_cols,
    region_col="region",
    sub_col="sub_id",
    trial_col="trial_index",
    n_components=3,
    zscore_within_subject=True,
):
    """
    Pool trials across subjects within each region, but first optionally
    z-score each feature within subject-region to reduce between-subject scale dominance.

    Returns
    -------
    pca_results : dict
        pca_results[region] = {
            "pca": fitted PCA,
            "scores_df": dataframe with PC scores + metadata,
            "loadings_df": dataframe of loadings,
            "explained_variance_ratio": ...
        }
    """
    pca_results = {}

    for region, df_r in df.groupby(region_col, sort=False):
        blocks = []

        for sub, g in df_r.groupby(sub_col, sort=False):
            g = g.sort_values(trial_col).copy()
            X = g[feature_cols].apply(pd.to_numeric, errors="coerce")

            if zscore_within_subject:
                X = X.apply(lambda s: _safe_zscore(s.values), axis=0, result_type="expand")
                X.columns = feature_cols

            keep = np.isfinite(X.values).sum(axis=1) == len(feature_cols)
            X = X.loc[keep]
            g = g.loc[keep]

            if len(X) < 3:
                continue

            tmp = g[[sub_col, region_col, trial_col]].copy()
            for c in feature_cols:
                tmp[c] = X[c].values
            blocks.append(tmp)

        if len(blocks) == 0:
            pca_results[region] = None
            continue

        df_pool = pd.concat(blocks, axis=0, ignore_index=True)
        X_pool = df_pool[feature_cols].values

        pca = PCA(n_components=min(n_components, len(feature_cols)))
        scores = pca.fit_transform(X_pool)

        score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
        scores_df = df_pool[[sub_col, region_col, trial_col]].copy()
        for i, c in enumerate(score_cols):
            scores_df[c] = scores[:, i]

        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=feature_cols,
            columns=score_cols
        )

        pca_results[region] = {
            "pca": pca,
            "scores_df": scores_df,
            "loadings_df": loadings_df,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "feature_cols": feature_cols,
        }

    return pca_results


def plot_pca_loadings_and_scores(
    pca_results,
    pc_x="PC1",
    pc_y="PC2",
    pcs=(1, 2, 3),
    trial_col="trial_index",
    sub_col="sub_id",
    figsize=(8, 3),
):
    """
    For each region:
      left  = loading scatter (PC1 vs PC2)
      right = multi-PC score-vs-trial with SEM
    """

    cmap = plt.get_cmap("tab10")

    for region, res in pca_results.items():
        if res is None:
            continue

        loadings = res["loadings_df"]
        scores_df = res["scores_df"]

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # ========================
        # loadings
        # ========================
        ax = axes[0]
        ax.axhline(0, lw=1)
        ax.axvline(0, lw=1)

        for feat in loadings.index:
            x = loadings.loc[feat, pc_x]
            y = loadings.loc[feat, pc_y]
            ax.scatter(x, y, s=50)
            ax.text(x, y, feat, fontsize=6)

        evr = res["explained_variance_ratio"]
        pcx_idx = int(pc_x.replace("PC", "")) - 1
        pcy_idx = int(pc_y.replace("PC", "")) - 1

        ax.set_title(
            f"{region} loadings\n"
            f"{pc_x} ({evr[pcx_idx]*100:.1f}%), {pc_y} ({evr[pcy_idx]*100:.1f}%)"
        )
        ax.set_xlabel(pc_x)
        ax.set_ylabel(pc_y)

        # ========================
        # multi-PC score vs trial
        # ========================
        ax = axes[1]

        for i, pc in enumerate(pcs):
            pc_name = f"PC{pc}"
            color = cmap(i)

            grp = scores_df.groupby(trial_col)[pc_name]
            mean = grp.mean()
            sem = grp.sem()

            ax.plot(mean.index, mean.values, lw=2, label=pc_name, color=color)
            ax.fill_between(
                mean.index,
                mean.values - sem.values,
                mean.values + sem.values,
                alpha=0.2,
                color=color,
            )

        ax.axhline(0, lw=1, ls="--")
        ax.set_title(f"{region} PC scores vs trial")
        ax.set_xlabel(trial_col)
        ax.set_ylabel("Score")
        ax.set_xticks(np.arange(0,65,5))
        ax.grid()

        ax.legend(bbox_to_anchor=(1.0,0),loc="lower left",frameon=False)

        plt.show()


# =========================================================
# 4) amplitude-vs-latency CCA / PLS
# =========================================================
def run_multiblock_synergy_analysis(
    df,
    x_cols,
    y_cols,
    region_col="region",
    sub_col="sub_id",
    trial_col="trial_index",
    method="cca",   # "cca" or "pls"
    n_components=2,
    zscore_within_subject=True,
    min_n_rows_per_subject=5,
    x_name="X",
    y_name="Y",
):
    """
    Run CCA or PLS between any two feature blocks within each region on pooled subject-trials.

    Parameters
    ----------
    df : DataFrame
    x_cols : list[str]
        Columns for block X.
    y_cols : list[str]
        Columns for block Y.
    region_col, sub_col, trial_col : str
        Metadata columns.
    method : str
        "cca" or "pls".
    n_components : int
        Number of latent dimensions.
    zscore_within_subject : bool
        Whether to z-score each feature within subject (within region).
    min_n_rows_per_subject : int
        Minimum valid rows required for a subject to contribute.
    x_name, y_name : str
        Names of the two blocks, used in output.

    Returns
    -------
    results[region] = {
        "x_cols": ...,
        "y_cols": ...,
        "x_name": ...,
        "y_name": ...,
        "model": fitted model,
        "scores_df": latent scores + metadata,
        "x_weights": DataFrame,
        "y_weights": DataFrame,
        "pair_corrs": list,
        "method": str,
        "n_components": int,
        "n_rows": int,
        "n_subjects": int,
    }
    """
    x_cols = list(x_cols)
    y_cols = list(y_cols)

    if len(x_cols) == 0 or len(y_cols) == 0:
        raise ValueError("x_cols and y_cols must both be non-empty.")

    missing_x = [c for c in x_cols if c not in df.columns]
    missing_y = [c for c in y_cols if c not in df.columns]
    if missing_x or missing_y:
        raise ValueError(
            f"Missing columns. x missing={missing_x}, y missing={missing_y}"
        )

    results = {}

    for region, df_r in df.groupby(region_col, sort=False):
        blocks = []

        for sub, g in df_r.groupby(sub_col, sort=False):
            g = g.sort_values(trial_col).copy()

            X = g[x_cols].apply(pd.to_numeric, errors="coerce")
            Y = g[y_cols].apply(pd.to_numeric, errors="coerce")

            if zscore_within_subject:
                X = X.apply(lambda s: _safe_zscore(s.values), axis=0, result_type="expand")
                X.columns = x_cols
                Y = Y.apply(lambda s: _safe_zscore(s.values), axis=0, result_type="expand")
                Y.columns = y_cols

            keep = (
                np.isfinite(X.values).sum(axis=1) == len(x_cols)
            ) & (
                np.isfinite(Y.values).sum(axis=1) == len(y_cols)
            )

            X = X.loc[keep]
            Y = Y.loc[keep]
            g = g.loc[keep]

            if len(X) < min_n_rows_per_subject:
                continue

            tmp = g[[sub_col, region_col, trial_col]].copy()
            for c in x_cols:
                tmp[c] = X[c].values
            for c in y_cols:
                tmp[c] = Y[c].values
            blocks.append(tmp)

        if len(blocks) == 0:
            results[region] = None
            continue

        df_pool = pd.concat(blocks, axis=0, ignore_index=True)
        X_pool = df_pool[x_cols].values
        Y_pool = df_pool[y_cols].values

        n_comp = min(n_components, X_pool.shape[1], Y_pool.shape[1])
        if n_comp < 1:
            results[region] = None
            continue

        if method.lower() == "cca":
            model = CCA(n_components=n_comp, max_iter=2000)
            X_scores, Y_scores = model.fit_transform(X_pool, Y_pool)
            x_weights = pd.DataFrame(
                model.x_weights_,
                index=x_cols,
                columns=[f"LV{i+1}" for i in range(n_comp)],
            )
            y_weights = pd.DataFrame(
                model.y_weights_,
                index=y_cols,
                columns=[f"LV{i+1}" for i in range(n_comp)],
            )

        elif method.lower() == "pls":
            model = PLSRegression(n_components=n_comp)
            model.fit(X_pool, Y_pool)
            X_scores = model.x_scores_
            Y_scores = model.y_scores_
            x_weights = pd.DataFrame(
                model.x_weights_,
                index=x_cols,
                columns=[f"LV{i+1}" for i in range(n_comp)],
            )
            y_weights = pd.DataFrame(
                model.y_weights_,
                index=y_cols,
                columns=[f"LV{i+1}" for i in range(n_comp)],
            )
        else:
            raise ValueError("method must be 'cca' or 'pls'")

        score_cols_x = [f"{x_name}_LV{i+1}" for i in range(n_comp)]
        score_cols_y = [f"{y_name}_LV{i+1}" for i in range(n_comp)]

        scores_df = df_pool[[sub_col, region_col, trial_col]].copy()
        for i, c in enumerate(score_cols_x):
            scores_df[c] = X_scores[:, i]
        for i, c in enumerate(score_cols_y):
            scores_df[c] = Y_scores[:, i]

        pair_corrs = []
        for i in range(n_comp):
            x_i = X_scores[:, i]
            y_i = Y_scores[:, i]
            if np.all(np.isfinite(x_i)) and np.all(np.isfinite(y_i)):
                r = np.corrcoef(x_i, y_i)[0, 1]
            else:
                r = np.nan
            pair_corrs.append(r)

        results[region] = {
            "x_cols": x_cols,
            "y_cols": y_cols,
            "x_name": x_name,
            "y_name": y_name,
            "model": model,
            "scores_df": scores_df,
            "x_weights": x_weights,
            "y_weights": y_weights,
            "pair_corrs": pair_corrs,
            "method": method.lower(),
            "n_components": n_comp,
            "n_rows": len(df_pool),
            "n_subjects": df_pool[sub_col].nunique(),
        }

    return results

def run_named_multiblock_synergy_analysis(
    df,
    block_dict,
    x_block,
    y_block,
    region_col="region",
    sub_col="sub_id",
    trial_col="trial_index",
    method="cca",
    n_components=2,
    zscore_within_subject=True,
    min_n_rows_per_subject=5,
):
    """
    Wrapper around run_multiblock_synergy_analysis using named blocks.

    Example
    -------
    block_dict = {
        "amp": amp_cols,
        "lat": lat_cols,
        "freq": freq_cols,
    }
    """
    if x_block not in block_dict:
        raise ValueError(f"x_block='{x_block}' not found in block_dict")
    if y_block not in block_dict:
        raise ValueError(f"y_block='{y_block}' not found in block_dict")

    return run_multiblock_synergy_analysis(
        df=df,
        x_cols=block_dict[x_block],
        y_cols=block_dict[y_block],
        region_col=region_col,
        sub_col=sub_col,
        trial_col=trial_col,
        method=method,
        n_components=n_components,
        zscore_within_subject=zscore_within_subject,
        min_n_rows_per_subject=min_n_rows_per_subject,
        x_name=x_block,
        y_name=y_block,
    )


def plot_multiblock_results(
    multiblock_results,
    lv=1,
    trial_col="trial_index",
    figsize=(6, 5),
):
    import matplotlib.gridspec as gridspec

    lv_idx = lv - 1

    for region, res in multiblock_results.items():
        if res is None:
            continue

        xw = res["x_weights"].iloc[:, lv_idx]
        yw = res["y_weights"].iloc[:, lv_idx]
        scores_df = res["scores_df"]

        x_name = res.get("x_name", "X")
        y_name = res.get("y_name", "Y")

        x_col = f"{x_name}_LV{lv}"
        y_col = f"{y_name}_LV{lv}"

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[0, 1])
        ax_s = fig.add_subplot(gs[1, :])

        ax_x.bar(range(len(xw)), xw.values)
        ax_x.set_xticks(range(len(xw)))
        ax_x.set_xticklabels(xw.index, rotation=-45, fontsize=8, ha="left")
        ax_x.set_title(f"{region} {x_name} weights ({res['method'].upper()} LV{lv})")

        ax_y.bar(range(len(yw)), yw.values)
        ax_y.set_xticks(range(len(yw)))
        ax_y.set_xticklabels(yw.index, rotation=-45, fontsize=8, ha="left")
        ax_y.set_title(f"{region} {y_name} weights ({res['method'].upper()} LV{lv})")

        grp = scores_df.groupby(trial_col)[[x_col, y_col]]
        mean = grp.mean()
        sem = grp.sem()

        ax_s.plot(mean.index, mean[x_col], label=x_col, lw=2)
        ax_s.plot(mean.index, mean[y_col], label=y_col, lw=2)

        ax_s.fill_between(
            mean.index,
            mean[x_col] - sem[x_col],
            mean[x_col] + sem[x_col],
            alpha=0.2,
        )
        ax_s.fill_between(
            mean.index,
            mean[y_col] - sem[y_col],
            mean[y_col] + sem[y_col],
            alpha=0.2,
        )

        ax_s.axhline(0, lw=1, ls="--")
        ax_s.set_title(
            f"{region} LV{lv} score-vs-trial\npaired corr={res['pair_corrs'][lv_idx]:.3f}"
        )
        ax_s.set_xlabel(trial_col)
        ax_s.legend(frameon=False)

        plt.show()


# =========================================================
# 5) one-stop wrapper
# =========================================================

def run_erp_synergy_pipeline(
    df,
    feature_cols,
    region_col="region",
    sub_col="sub_id",
    trial_col="trial_index",
    zscore_within_subject=True,
    pca_n_components=3,
    multiblock_method="cca",   # "cca" or "pls"
    multiblock_n_components=2,
    block_dict=None,
    multiblock_pairs=None,
):
    """
    Full pipeline:
      1) residualization
      2) group-level correlation on raw
      3) group-level correlation on residualized
      4) PCA on residualized features
      5) multiblock synergy on residualized features

    Parameters
    ----------
    block_dict : dict or None
        Example:
        {
            "amp": [...],
            "lat": [...],
            "freq": [...],
        }
        If None, infer from residualized feature names.

    multiblock_pairs : list[tuple[str, str]] or None
        Example:
            [("amp", "lat"), ("freq", "lat")]
        If None, run all pairwise combinations found in block_dict.

    Returns
    -------
    out : dict
        Includes keys like:
            out["multiblock_amp_lat_res"]
            out["multiblock_freq_lat_res"]
    """
    # ---- residualization
    df_resid, beta_df = residualize_features_within_subject(
        df=df,
        feature_cols=feature_cols,
        region_col=region_col,
        sub_col=sub_col,
        trial_col=trial_col,
        suffix="_resid",
        zscore_within_subject=zscore_within_subject,
    )
    resid_cols = [f"{c}_resid" for c in feature_cols]

    # ---- correlation
    corr_raw = compute_group_level_correlation(
        df=df,
        feature_cols=feature_cols,
        region_col=region_col,
        sub_col=sub_col,
    )

    corr_resid = compute_group_level_correlation(
        df=df_resid,
        feature_cols=resid_cols,
        region_col=region_col,
        sub_col=sub_col,
    )

    # ---- PCA on residuals
    pca_res = run_group_pca_by_region(
        df=df_resid,
        feature_cols=resid_cols,
        region_col=region_col,
        sub_col=sub_col,
        trial_col=trial_col,
        n_components=pca_n_components,
        zscore_within_subject=False,
    )

    # ---- build block dict on residualized columns
    if block_dict is None:
        resid_block_dict = _make_named_feature_blocks(resid_cols)
    else:
        resid_block_dict = {}
        for block_name, cols in block_dict.items():
            resid_block_dict[block_name] = [
                c if c.endswith("_resid") else f"{c}_resid"
                for c in cols
            ]

    # only keep non-empty valid blocks
    resid_block_dict = {
        k: [c for c in v if c in df_resid.columns]
        for k, v in resid_block_dict.items()
    }
    resid_block_dict = {
        k: v for k, v in resid_block_dict.items() if len(v) > 0
    }

    # ---- which block pairs to run
    if multiblock_pairs is None:
        block_names = list(resid_block_dict.keys())
        multiblock_pairs = list(combinations(block_names, 2))

    # ---- run named multiblock synergies
    multiblock_out = {}

    for x_block, y_block in multiblock_pairs:
        if x_block not in resid_block_dict or y_block not in resid_block_dict:
            continue
        if len(resid_block_dict[x_block]) == 0 or len(resid_block_dict[y_block]) == 0:
            continue

        res = run_named_multiblock_synergy_analysis(
            df=df_resid,
            block_dict=resid_block_dict,
            x_block=x_block,
            y_block=y_block,
            region_col=region_col,
            sub_col=sub_col,
            trial_col=trial_col,
            method=multiblock_method,
            n_components=multiblock_n_components,
            zscore_within_subject=False,
        )

        key = f"multiblock_{x_block}_{y_block}_res"
        multiblock_out[key] = res

    # ---- backward compatibility
    # if amp-lat exists, keep the old generic key too
    generic_multiblock_res = None
    if "multiblock_amp_lat_res" in multiblock_out:
        generic_multiblock_res = multiblock_out["multiblock_amp_lat_res"]

    out = {
        "df_resid": df_resid,
        "beta_df": beta_df,
        "corr_raw": corr_raw,
        "corr_resid": corr_resid,
        "pca_res": pca_res,
        "feature_cols": feature_cols,
        "resid_cols": resid_cols,
        "block_dict": resid_block_dict,
        "multiblock_res": generic_multiblock_res,
    }

    out.update(multiblock_out)

    return out