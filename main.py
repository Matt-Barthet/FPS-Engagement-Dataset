import importlib  
data_loading = importlib.import_module("1-data-loading")
data_processing = importlib.import_module("2-data-preprocessing")
gold_standard = importlib.import_module("3-gold-standard")

if __name__ == "__main__":

    print("\rStep 1/4: Combining raw CSV files into Visual/Audio/Engagement CSVs...", end="")
    keep_latest_session = True 
    data_loading.execute(keep_latest_session)    
    print("\rStep 1/4: Combining raw CSV files into Visual/Audio/Engagement CSVs... done!")

    print("\rStep 2/4: Creating session dictionary, interpolating and smoothening signals...", end="")
    time_window_size_ms = 1000 
    normalize_signals = True 
    minimum_changes = 1
    moving_average_window = 1
    data_processing.execute(normalize_signals, time_window_size_ms, minimum_changes, moving_average_window)
    print("\rStep 2/4: Creating session dictionary, interpolating and smoothening signals... done!",)

    print("\rStep 3/4: Calculating gold standard signals...", end="")
    gold_standard.execute()
    print("\rStep 3/4: Calculating gold standard signals... done!")
