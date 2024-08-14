
from datetime import datetime
from os import getcwd

def get_date():
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    weights_directory_path = getcwd() + "/weights/" + date_time_str 
    return date_time_str