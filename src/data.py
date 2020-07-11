from util import Timer
import pandas as pd
import ujson as json
from config import *
import os

###################
# new
def load_city_map():
    with open(os.path.join(data_dir, "city_map.json"), 'r', encoding='utf-8') as infile:
        city_map = json.load(infile)
    return city_map

def load_data(phrase="train"):
    data = pd.read_parquet(os.path.join(data_dir, "{}.parquet".format(phrase)))
    return data

###########################
# old
"""
def get_id_city_key(d):
    id_key, city_key = None, None
    for k in id_city_key.keys():
        if k in d:
            id_key = id_city_key[k][0]
            city_key = id_city_key[k][1]
    if id_key == None or city_key == None:
        print("Unable to find tweet ID and city key; json =", d)
        raise SystemExit

    return id_key, city_key

def load_data(filename, feature_extractor, conf):
    data = []
    id_key = None
    with open(filename, "r", encoding='utf-8') as infile:
        for count, line in enumerate(infile):
            d = json.loads(line)
            if id_key == None:
                id_key, _ = get_id_city_key(d)
            data.append(feature_extractor(d, id_key))
            if count%100000 == 0 and conf.verbose:
                print("\rprocessing {} lines   ".format(count), end="")
    print()
    return data

# only city right now
# TODO: modify city_key for (lon, lan)
def load_label(filename, conf, lamb=lambda x:x):
    label = {}
    id_key, city_key = None, None
     
    with open(filename, 'r', encoding='utf-8') as infile:
        for count, line in enumerate(infile):
            d = json.loads(line.strip())
            if id_key is None:
                id_key, city_key = get_id_city_key(d)
            label[d[id_key]] = lamb(d[city_key])
            if count % 100000 == 0 and conf.verbose:
                print("\rprocessing {} lines     ".format(count), end="")
    print()
    return label
"""
if __name__ == "__main__":
    pass
