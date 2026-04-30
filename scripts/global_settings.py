roi_dict = {
    "auditory_fc": ["F3", "FZ", "F4", "FC1", "FC2", "Cz"],
    "tactile_central": ["C3", "Cz", "C4", "CP3", "CPZ", "CP4"],
    "late_cp": ["CP1", "CPZ", "CP2", "P1", "P2", "POz"],
    "all_rois": ["F3", "FZ", "F4", "FC1", "FC2", "Cz", "C3", "C4", "CP3", "CPZ", "CP4", "CP1", "CP2", "P1", "P2", "POz", "T7", "T8"]
}

roi_region_dict = {
    "auditory_fc": {
        "F": ["F3", "FZ", "F4"],
        "FC": ["FC1", "FC2"],
        "Cz": ["Cz"]
    },
    "tactile_central": {
        "C": ["C3", "Cz", "C4"],
        "CP": ["CP3", "CPZ", "CP4"]
    },
    "late_cp": {
        "CP": ["CP1", "CPZ", "CP2"],
        "P": ["P1", "P2"],
        "PO": ["POz"]
    },
    "all_rois": {
        "F": ["F3", "FZ", "F4"],
        "FC": ["FC1", "FC2"],
        "C": ["C3", "Cz", "C4"],
        "CP": ["CP3", "CPZ", "CP4"],
        "P": ["P1", "P2"],
        "PO": ["POz"],
        "T": ["T7", "T8"],
    },
    "all_rois_flat": {
        "F3": ["F3"],
        "FZ": ["FZ"],
        "F4": ["F4"],
        "FC1": ["FC1"],
        "FC2": ["FC2"],
        "C3": ["C3"],
        "Cz": ["Cz"],
        "C4": ["C4"],
        "CP3": ["CP3"],
        "CPZ": ["CPZ"],
        "CP4": ["CP4"],
        "P1": ["P1"],
        "P2": ["P2"],
        "POz": ["POz"],
        "T7": ["T7"],
        "T8": ["T8"],
    },
}

ROLLING_WINDOW = 3

auditory_search = {
    "P50": [30, 100],
    "N100":  [80, 160],
    "P200":  [150, 250],
    "N200":  [180, 320],
    "P300":  [250, 500],
}

tactile_search = {
    "P50": [20, 70],
    "N100":  [50, 120],
    "P200":  [100, 200],
    "N200":  [130, 260],
    "P300":  [220, 420],
}

p3_search = {
    "A_P50":   [30, 100],
    "A_N100":  [80, 160],
    "A_P200":  [150, 250],
    "A_N200":  [180, 320],
    "A_P300":  [250, 500],
    "T_P50":   [20 + 500, 70 + 500],
    "T_N100":  [50 + 500, 120 + 500],
    "T_P200":  [100 + 500, 200 + 500],
    "T_N200":  [130 + 500, 260 + 500],
    "T_P300":  [220 + 500, 420 + 500],
}


half_width_map= {
    "P50": 15,
    "N100":  25,
    "P200":  25,
    "N200":  25,
    "P300":  50,
}

p3_half_width_map={
    "A_P50":   15,
    "A_N100":  25,
    "A_P200":  25,
    "A_N200":  25,
    "A_P300":  50,
    "T_P50":   15,
    "T_N100":  25,
    "T_P200":  25,
    "T_N200":  25,
    "T_P300":  50,
}

tactile_polarity = {
    "P50": "pos",
    "N100":  "neg",
    "P200":  "pos",
    "N200":  "neg",
    "P300":  "pos",
}

auditory_polarity = {
    "P50": "pos",
    "N100":  "neg",
    "P200":  "pos",
    "N200":  "neg",
    "P300":  "pos",
}

p3_polarity = {
    "A_P50":   "pos",
    "A_N100":  "neg",
    "A_P200":  "pos",
    "A_N200":  "neg",
    "A_P300":  "pos",
    "T_P50":   "pos",
    "T_N100":  "neg",
    "T_P200":  "pos",
    "T_N200":  "neg",
    "T_P300":  "pos",
}

SEARCH_DICT = {
    "BLT":{
        "search_windows": tactile_search,
        "polarity_map": tactile_polarity,
        "half_width_map": half_width_map,
    },
    "BLA":{
        "search_windows": auditory_search,
        "polarity_map": auditory_polarity,
        "half_width_map": half_width_map,
    },
    "P3":{
        "search_windows": p3_search,
        "polarity_map": p3_polarity,
        "half_width_map": p3_half_width_map,
    }
}

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

erp_feature_cols ={
    "BLA": [
        "P50_mean_amp", "P50_peak_lat_ms",
        "N100_mean_amp", "N100_peak_lat_ms",
        "P200_mean_amp", "P200_peak_lat_ms",
        "N200_mean_amp", "N200_peak_lat_ms",
        "P300_mean_amp", "P300_peak_lat_ms",
    ],
    "BLT": [
        "P50_mean_amp", "P50_peak_lat_ms",
        "N100_mean_amp", "N100_peak_lat_ms",
        "P200_mean_amp", "P200_peak_lat_ms",
        "N200_mean_amp", "N200_peak_lat_ms",
        "P300_mean_amp", "P300_peak_lat_ms",
    ],
    "P3": [
        "A_P50_mean_amp", "A_P50_peak_lat_ms",
        "A_N100_mean_amp", "A_N100_peak_lat_ms",
        "A_P200_mean_amp", "A_P200_peak_lat_ms",
        "A_N200_mean_amp", "A_N200_peak_lat_ms",
        "A_P300_mean_amp", "A_P300_peak_lat_ms",
        "T_P50_mean_amp", "T_P50_peak_lat_ms",
        "T_N100_mean_amp", "T_N100_peak_lat_ms",
        "T_P200_mean_amp", "T_P200_peak_lat_ms",
        "T_N200_mean_amp", "T_N200_peak_lat_ms",
        "T_P300_mean_amp", "T_P300_peak_lat_ms",
    ]
}

# erp_feature_cols ={
#     "BLA": [
#         "P50_mean_amp", "P50_peak_amp", "P50_peak_lat_ms",
#         "N100_mean_amp", "N100_peak_amp", "N100_peak_lat_ms",
#         "P200_mean_amp", "P200_peak_amp", "P200_peak_lat_ms",
#         "N200_mean_amp", "N200_peak_amp", "N200_peak_lat_ms",
#         "P300_mean_amp", "P300_peak_amp", "P300_peak_lat_ms",
#     ],
#     "BLT": [
#         "P50_mean_amp", "P50_peak_amp", "P50_peak_lat_ms",
#         "N100_mean_amp", "N100_peak_amp", "N100_peak_lat_ms",
#         "P200_mean_amp", "P200_peak_amp", "P200_peak_lat_ms",
#         "N200_mean_amp", "N200_peak_amp", "N200_peak_lat_ms",
#         "P300_mean_amp", "P300_peak_amp", "P300_peak_lat_ms",
#     ],
#     "P3": [
#         "A_P50_mean_amp", "A_P50_peak_amp", "A_P50_peak_lat_ms",
#         "A_N100_mean_amp", "A_N100_peak_amp", "A_N100_peak_lat_ms",
#         "A_P200_mean_amp", "A_P200_peak_amp", "A_P200_peak_lat_ms",
#         "A_N200_mean_amp", "A_N200_peak_amp", "A_N200_peak_lat_ms",
#         "A_P300_mean_amp", "A_P300_peak_amp", "A_P300_peak_lat_ms",
#         "T_P50_mean_amp", "T_P50_peak_amp", "T_P50_peak_lat_ms",
#         "T_N100_mean_amp", "T_N100_peak_amp", "T_N100_peak_lat_ms",
#         "T_P200_mean_amp", "T_P200_peak_amp", "T_P200_peak_lat_ms",
#         "T_N200_mean_amp", "T_N200_peak_amp", "T_N200_peak_lat_ms",
#         "T_P300_mean_amp", "T_P300_peak_amp", "T_P300_peak_lat_ms",
#     ]
# } 