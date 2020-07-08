from util import Timer
from pymongo import MongoClient

try:
    import ujson as json
except:
    import json

id_city_key = {
    "hashed_tweet_id": ("hashed_tweet_id", "city"), 
    "tweet_id": ("tweet_id", "tweet_city"), 
    "id_str": ("id_str", "tweet_city")
}

# TODO: not finished
def mongo_load_data(phase, query, projection):
    mongo = MongoClient("localhost")["twitter_location"][phase]
    res = mongo.find(query, projection=projection)
    return [r for r in res]

def load_city_map():
    mongo = MongoClient("localhost")["twitter_location"]["city_map"]
    city = {
        r["city"]:(r["lat_value"], r["lon_value"]) 
        for r in mongo["city_map"].find(
            {}, 
            projection={"_id":False, "city":True, "lat_value":True, "lon_value":True}
        )
    }
    return city

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

if __name__ == "__main__":
    pass
