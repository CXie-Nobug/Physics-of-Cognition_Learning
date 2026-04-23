import numpy as np
import pandas as pd
from scipy.signal import welch


def extract_erp_and_freq_features(
    all_sub_data,
    subject_ids,
    selected_channels,
    component_windows=None,
    res_aud=None,
    freq_windows=((0, 500), (500, 1000)),   # ms
    freq_dict=None,
    b_relative_band_power=True,
    sfreq=None,
    welch_nperseg=None,
):
    """
    Extract ERP features and frequency-band power features for each subject/trial.

    Parameters
    ----------
    all_sub_data : dict
        all_sub_data[sub_id] should contain:
            - "data_bc": ndarray, shape (n_trials, n_channels, n_times)
            - "times": ndarray, shape (n_times,), in seconds
            - optionally "sfreq"
    subject_ids : iterable
        Subject IDs to process.
    selected_channels : list
        Channel names aligned with axis=1 of data_bc.
    component_windows : dict or None
        Dict like:
        {
            "P1": {"tmin_s": 0.05, "tmax_s": 0.10, "polarity": "pos"},
            ...
        }
        If None, will build from res_aud.
    res_aud : pd.DataFrame or None
        Used only when component_windows is None.
        Expect columns: final_start_ms, final_end_ms, polarity
    freq_windows : list/tuple of tuple
        Time windows in ms, e.g. [(0, 500), (500, 1000)]
    freq_dict : dict or None
        Frequency bands, e.g.
        {
            "below_delta": (0, 0.5),
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        }
    b_relative_band_power : bool
        If True, return relative band power ratio within each freq_window,
        normalized by the sum of all bands in freq_dict for the same trial/channel/window.
    sfreq : float or None
        Sampling frequency. If None, try all_sub_data[sub_id]["sfreq"], else infer from times.
    welch_nperseg : int or None
        nperseg for scipy.signal.welch. If None, auto choose based on window length.

    Returns
    -------
    results : dict
        {
            "component_windows": ...,
            "erp_features_df": per-channel ERP features,
            "erp_roi_features_df": ROI-averaged ERP features,
            "freq_features_df": per-channel frequency features (wide format),
            "freq_roi_features_df": ROI-averaged frequency features,
        }
    """
    if component_windows is None:
        if res_aud is None:
            raise ValueError("Either component_windows or res_aud must be provided.")
        component_windows = {
            comp: {
                "tmin_s": row["final_start_ms"] / 1000.0,
                "tmax_s": row["final_end_ms"] / 1000.0,
                "polarity": row["polarity"],
            }
            for comp, row in res_aud.iterrows()
        }

    if freq_dict is None:
        freq_dict = {
            "below_delta": (0, 0.5),
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        }

    all_erp_features_df_list = []
    all_erp_roi_features_df_list = []
    all_freq_features_df_list = []
    all_freq_roi_features_df_list = []

    for sub_id in subject_ids:
        sub_pack = all_sub_data[sub_id]
        data_bc = sub_pack["data_bc"]   # (n_trials, n_channels, n_times)
        times = sub_pack["times"]       # seconds
        ch_names = selected_channels

        n_trials, n_channels, n_times = data_bc.shape

        # ---- sampling rate
        if sfreq is not None:
            sfreq_sub = sfreq
        elif "sfreq" in sub_pack:
            sfreq_sub = sub_pack["sfreq"]
        else:
            dt = np.nanmedian(np.diff(times))
            sfreq_sub = 1.0 / dt

        # =========================================================
        # Part 1. ERP features
        # =========================================================
        window_masks = {}
        for comp, info in component_windows.items():
            mask = (times >= info["tmin_s"]) & (times <= info["tmax_s"])
            window_masks[comp] = mask

        feat_mean = {}
        feat_peak_amp = {}
        feat_peak_lat_ms = {}

        for comp, mask in window_masks.items():
            if mask.sum() == 0:
                raise ValueError(f"ERP window for {comp} has no samples in times.")

            feat_mean[comp] = np.nanmean(data_bc[:, :, mask], axis=-1)

            polarity = component_windows[comp]["polarity"]
            times_win = times[mask]
            win_data = data_bc[:, :, mask]

            if polarity == "pos":
                peak_ix = np.nanargmax(win_data, axis=-1)
            elif polarity == "neg":
                peak_ix = np.nanargmin(win_data, axis=-1)
            else:
                raise ValueError(f"Unknown polarity for {comp}: {polarity}")

            peak_amp = np.take_along_axis(
                win_data,
                peak_ix[..., None],
                axis=-1
            ).squeeze(-1)

            peak_lat_ms = times_win[peak_ix] * 1000.0

            feat_peak_amp[comp] = peak_amp
            feat_peak_lat_ms[comp] = peak_lat_ms

        # ---- per-channel ERP df
        erp_rows = []
        for t in range(n_trials):
            for c, ch in enumerate(ch_names):
                row = {
                    "sub_id": sub_id,
                    "trial_index": t + 1,
                    "channel": ch,
                }
                for comp in component_windows.keys():
                    row[f"{comp}_mean_amp"] = feat_mean[comp][t, c]
                    row[f"{comp}_peak_amp"] = feat_peak_amp[comp][t, c]
                    row[f"{comp}_peak_lat_ms"] = feat_peak_lat_ms[comp][t, c]
                erp_rows.append(row)

        erp_features_df_sub = pd.DataFrame(erp_rows)
        all_erp_features_df_list.append(erp_features_df_sub)

        # ---- ROI ERP df
        erp_roi_rows = []
        for t in range(n_trials):
            row = {
                "sub_id": sub_id,
                "trial_index": t + 1,
            }
            for comp in component_windows.keys():
                row[f"{comp}_mean_amp_roi"] = np.nanmean(feat_mean[comp][t])
                row[f"{comp}_peak_amp_roi"] = np.nanmean(feat_peak_amp[comp][t])
                row[f"{comp}_peak_lat_ms_roi"] = np.nanmean(feat_peak_lat_ms[comp][t])
            erp_roi_rows.append(row)

        erp_roi_features_df_sub = pd.DataFrame(erp_roi_rows)
        all_erp_roi_features_df_list.append(erp_roi_features_df_sub)

        # =========================================================
        # Part 2. Frequency-band power features
        # =========================================================
        # 改动点：
        # per-channel freq_features_df 不再是 long format
        # 而是每个 trial × channel 一行，所有 freq window × band 直接展开成列
        freq_rows = []
        freq_roi_rows = []

        for t in range(n_trials):
            roi_row = {
                "sub_id": sub_id,
                "trial_index": t + 1,
            }

            # temporary container for ROI aggregation
            roi_band_vals = {}

            # 先给每个 channel 建一个宽表 row
            channel_rows = []
            for c, ch in enumerate(ch_names):
                channel_rows.append({
                    "sub_id": sub_id,
                    "trial_index": t + 1,
                    "channel": ch,
                })

            for fw_idx, (w_start_ms, w_end_ms) in enumerate(freq_windows):
                w_start_s = w_start_ms / 1000.0
                w_end_s = w_end_ms / 1000.0
                mask = (times >= w_start_s) & (times <= w_end_s)

                if mask.sum() < 2:
                    for c, ch in enumerate(ch_names):
                        for band_name in freq_dict.keys():
                            col_name = f"fw{fw_idx}_{band_name}_power"
                            channel_rows[c][col_name] = np.nan

                    for band_name in freq_dict.keys():
                        roi_band_vals[(fw_idx, band_name)] = np.nan
                    continue

                for c, ch in enumerate(ch_names):
                    sig = data_bc[t, c, mask]

                    if np.all(np.isnan(sig)):
                        freqs = None
                        psd = None
                    else:
                        sig = np.asarray(sig, dtype=float)
                        if np.isnan(sig).any():
                            good = ~np.isnan(sig)
                            if good.sum() < 2:
                                freqs = None
                                psd = None
                            else:
                                sig = np.interp(
                                    np.arange(len(sig)),
                                    np.where(good)[0],
                                    sig[good]
                                )
                                nperseg_use = welch_nperseg or min(len(sig), 256)
                                freqs, psd = welch(sig, fs=sfreq_sub, nperseg=nperseg_use)
                        else:
                            nperseg_use = welch_nperseg or min(len(sig), 256)
                            freqs, psd = welch(sig, fs=sfreq_sub, nperseg=nperseg_use)

                    band_powers = {}
                    if freqs is None or psd is None:
                        for band_name in freq_dict.keys():
                            band_powers[band_name] = np.nan
                    else:
                        for band_name, (fmin, fmax) in freq_dict.items():
                            band_mask = (freqs >= fmin) & (freqs < fmax)
                            if band_mask.sum() == 0:
                                band_powers[band_name] = np.nan
                            else:
                                band_powers[band_name] = np.trapz(psd[band_mask], freqs[band_mask])

                    if b_relative_band_power:
                        denom = np.nansum(list(band_powers.values()))
                        if denom > 0:
                            band_powers = {
                                k: v / denom if np.isfinite(v) else np.nan
                                for k, v in band_powers.items()
                            }
                        else:
                            band_powers = {k: np.nan for k in band_powers}

                    # 直接写成 wide columns
                    for band_name, val in band_powers.items():
                        col_name = f"fw{fw_idx}_{band_name}_power"
                        channel_rows[c][col_name] = val

                        key = (fw_idx, band_name)
                        roi_band_vals.setdefault(key, [])
                        roi_band_vals[key].append(val)

            # append per-channel wide rows
            freq_rows.extend(channel_rows)

            # ROI row: one row per trial, with all freq windows × bands展开成列
            for fw_idx, (w_start_ms, w_end_ms) in enumerate(freq_windows):
                for band_name in freq_dict.keys():
                    vals = roi_band_vals.get((fw_idx, band_name), [])
                    if isinstance(vals, list):
                        val = np.nanmean(vals) if len(vals) > 0 else np.nan
                    else:
                        val = vals
                    roi_row[f"fw{fw_idx}_{int(w_start_ms)}_{int(w_end_ms)}ms_{band_name}_power_roi"] = val

            freq_roi_rows.append(roi_row)

        freq_features_df_sub = pd.DataFrame(freq_rows)
        freq_roi_features_df_sub = pd.DataFrame(freq_roi_rows)

        all_freq_features_df_list.append(freq_features_df_sub)
        all_freq_roi_features_df_list.append(freq_roi_features_df_sub)

    erp_features_all_df = pd.concat(all_erp_features_df_list, ignore_index=True)
    erp_roi_features_all_df = pd.concat(all_erp_roi_features_df_list, ignore_index=True)
    freq_features_all_df = pd.concat(all_freq_features_df_list, ignore_index=True)
    freq_roi_features_all_df = pd.concat(all_freq_roi_features_df_list, ignore_index=True)

    return {
        "component_windows": component_windows,
        "erp_features_df": erp_features_all_df,
        "erp_roi_features_df": erp_roi_features_all_df,
        "freq_features_df": freq_features_all_df,
        "freq_roi_features_df": freq_roi_features_all_df,
    }