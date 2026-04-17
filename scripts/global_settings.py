roi_dict = {
    "auditory_fc": ["F3", "FZ", "F4", "FC1", "FC2", "Cz"],
    "tactile_central": ["C3", "Cz", "C4", "CP3", "CPZ", "CP4"],
    "late_cp": ["CP1", "CPZ", "CP2", "P1", "P2", "POz"],
    "all_rois": ["F3", "FZ", "F4", "FC1", "FC2", "Cz", "C3", "C4", "CP3", "CPZ", "CP4", "CP1", "CP2", "P1", "P2", "POz"]
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
        "PO": ["POz"]
    }
}

ROLLING_WINDOW = 5

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


half_width_map= {
    "P50": 15,
    "N100":  25,
    "P200":  25,
    "N200":  25,
    "P300":  50,
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

SEARCH_DICT = {
    "BLT":{
        "search_windows": tactile_search,
        "polarity_map": tactile_polarity,
    },
    "BLA":{
        "search_windows": auditory_search,
        "polarity_map": auditory_polarity,
    }
}