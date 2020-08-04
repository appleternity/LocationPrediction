from util import Timer
import pandas as pd
import ujson as json
from config import *
import os

def load_city_map():
    with open(os.path.join(data_dir, "city_map.json"), 'r', encoding='utf-8') as infile:
        city_map = json.load(infile)
    return city_map

def load_data(phrase="train"):
    data = pd.read_parquet(os.path.join(data_dir, "{}.parquet".format(phrase)))
    return data

if __name__ == "__main__":
    pass
