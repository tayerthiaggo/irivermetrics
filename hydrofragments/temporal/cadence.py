import pandas as pd

def detect_cadence(da):
    if "time" not in da.coords:
        return "unknown"
    
    time_index = pd.DatetimeIndex(da.get_index("time"))
    if len(time_index) < 2:
        return "unknown"
        
    diff = time_index[1] - time_index[0]
    if diff >= pd.Timedelta(days=27):
        return "monthly"
    return "submonthly"
