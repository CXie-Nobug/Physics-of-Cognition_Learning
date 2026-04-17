import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_peak_in_window(time_ms, signal, window_ms, polarity="pos"):
    """
    在给定时间窗内找 peak / trough

    Parameters
    ----------
    time_ms : array-like, shape [T]
    signal : array-like, shape [T]
    window_ms : [start_ms, end_ms]
    polarity : {"pos", "neg"}
        pos: 找最大值
        neg: 找最小值

    Returns
    -------
    out : dict
        {
            "peak_idx": int or None,
            "peak_time_ms": float or np.nan,
            "peak_amp": float or np.nan,
            "search_start_ms": float,
            "search_end_ms": float,
            "polarity": str,
        }
    """
    time_ms = np.asarray(time_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)

    start_ms, end_ms = window_ms
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)

    if not np.any(mask):
        return {
            "peak_idx": None,
            "peak_time_ms": np.nan,
            "peak_amp": np.nan,
            "search_start_ms": start_ms,
            "search_end_ms": end_ms,
            "polarity": polarity,
        }

    idx_local = np.where(mask)[0]
    seg = signal[idx_local]

    if polarity == "pos":
        best_local = np.nanargmax(seg)
    elif polarity == "neg":
        best_local = np.nanargmin(seg)
    else:
        raise ValueError("polarity must be 'pos' or 'neg'")

    peak_idx = idx_local[best_local]

    return {
        "peak_idx": int(peak_idx),
        "peak_time_ms": float(time_ms[peak_idx]),
        "peak_amp": float(signal[peak_idx]),
        "search_start_ms": float(start_ms),
        "search_end_ms": float(end_ms),
        "polarity": polarity,
    }


def _default_half_width_ms(peak_time_ms):
    """
    根据 peak latency 自动决定最终窗半宽
    """
    if peak_time_ms < 120:
        return 15.0
    elif peak_time_ms < 250:
        return 25.0
    else:
        return 40.0


def build_final_window_from_peak(
    peak_time_ms,
    half_width_ms=None,
    min_time_ms=None,
    max_time_ms=None,
):
    """
    由 peak time 生成最终分析窗
    """
    if np.isnan(peak_time_ms):
        return [np.nan, np.nan]

    if half_width_ms is None:
        half_width_ms = _default_half_width_ms(peak_time_ms)

    tw = [peak_time_ms - half_width_ms, peak_time_ms + half_width_ms]

    if min_time_ms is not None:
        tw[0] = max(tw[0], min_time_ms)
    if max_time_ms is not None:
        tw[1] = min(tw[1], max_time_ms)

    return [float(tw[0]), float(tw[1])]


def detect_erp_windows_from_grand_avg(
    time_ms,
    grand_avg,
    search_windows,
    onset_ms=0,
    polarity_map=None,
    half_width_map=None,
    min_time_ms=None,
    max_time_ms=None,
):
    """
    在 grand average 上自动检测 ERP 成分，并生成最终时间窗

    Parameters
    ----------
    time_ms : array-like, shape [T]
    grand_avg : array-like, shape [T]
    search_windows : dict
        例如:
        {
            "P50": [30, 80],
            "N1": [70, 140],
            "P2": [130, 220],
            "N2": [180, 300],
            "P3": [250, 450],
        }

    polarity_map : dict or None
        例如:
        {
            "P50": "pos",
            "N1": "neg",
            "P2": "pos",
            "N2": "neg",
            "P3": "pos",
        }
        如果不提供，默认全是 "pos"

    half_width_map : dict or None
        每个成分最终窗半宽（ms）
        例如:
        {
            "P50": 12,
            "N1": 15,
            "P2": 20,
            "N2": 25,
            "P3": 40,
        }
        不提供则自动按 peak latency 决定

    Returns
    -------
    res_df : pd.DataFrame
        index 为 component name
        columns:
            peak_idx
            peak_time_ms
            peak_amp
            search_start_ms
            search_end_ms
            polarity
            final_start_ms
            final_end_ms
            final_half_width_ms
    """
    time_ms = np.asarray(time_ms, dtype=float)
    grand_avg = np.asarray(grand_avg, dtype=float)

    if polarity_map is None:
        polarity_map = {k: "pos" for k in search_windows}

    rows = []
    for comp, tw in search_windows.items():
        polarity = polarity_map.get(comp, "pos")
        
        tw = [tw_i + onset_ms for tw_i in tw]
        
        peak_info = _find_peak_in_window(
            time_ms=time_ms,
            signal=grand_avg,
            window_ms=tw,
            polarity=polarity,
        )

        peak_time = peak_info["peak_time_ms"]

        if half_width_map is not None and comp in half_width_map:
            half_width = float(half_width_map[comp])
        else:
            half_width = _default_half_width_ms(peak_time) if not np.isnan(peak_time) else np.nan

        final_tw = build_final_window_from_peak(
            peak_time_ms=peak_time,
            half_width_ms=half_width if not np.isnan(half_width) else None,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
        )

        row = {
            "component": comp,
            **peak_info,
            "final_start_ms": final_tw[0],
            "final_end_ms": final_tw[1],
            "final_half_width_ms": half_width,
        }
        rows.append(row)

    res_df = pd.DataFrame(rows).set_index("component")
    return res_df


def plot_grand_avg_with_windows(
    time_ms,
    grand_avg,
    res_df,
    title="Grand Average ERP with Detected Windows",
    figsize=(10, 5),
    onset_ms=0,
):
    """
    画 grand average，并标注 peak 和 final windows
    """
    time_ms = np.asarray(time_ms, dtype=float)
    grand_avg = np.asarray(grand_avg, dtype=float)

    plt.figure(figsize=figsize)
    plt.plot(time_ms, grand_avg, linewidth=2, label="grand avg", color="black")
    plt.axvline(onset_ms, color="red", linestyle="-", alpha=1.0)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    y_min = np.nanmin(grand_avg)
    y_max = np.nanmax(grand_avg)
    y_span = y_max - y_min if y_max > y_min else 1.0

    for comp, row in res_df.iterrows():
        fs = row["final_start_ms"]
        fe = row["final_end_ms"]
        pt = row["peak_time_ms"]
        pa = row["peak_amp"]
        color = "orange" if row["polarity"] == "pos" else "cyan"

        if not np.isnan(fs) and not np.isnan(fe):
            plt.axvspan(fs, fe, alpha=0.15, color=color)

        if not np.isnan(pt) and not np.isnan(pa):
            plt.scatter([pt], [pa], s=60, color=color)
            plt.text(
                pt,
                pa + 0.03 * y_span,
                f"{comp}\n{pt:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xlim(-500+onset_ms, 1000+onset_ms)
    # plt.ylim(-1.5,1.5)
    plt.xticks(np.arange(-500+onset_ms, 1001+onset_ms, 100), rotation=30)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()