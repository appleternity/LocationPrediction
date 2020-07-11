import ujson as json
import os
import argparse
import pandas as pd
from twokenize import tokenizeRawTweetText 

label_keys = [
    ("tweet_city", "tweet_id", "tweet_latitude", "tweet_longitude"),
    ("city", "hashed_tweet_id", "lat", "lon")
]

def parse_arg():
    parser = argparse.ArgumentParser(description="Preprocessing Tweets and Labels.")
    parser.add_argument("--tweet_path", dest="tweet_path", help="path to the tweet corpus", type=str, required=True)
    parser.add_argument("--label_path", dest="label_path", help="path to the annotation", type=str, required=True)
    parser.add_argument("--output_path", dest="output_path", help="path to the output data", type=str, required=True)
    return parser.parse_args()

def preprocessing():
    arg = parse_arg()
    label_file = arg.label_path
    data_file = arg.tweet_path

    # city map
    #with open(os.path.join(data_dir, "city_map.json"), 'r', encoding='utf-8') as infile:
    #    city_map = json.load(infile)
    #return city_map

    print("loading label: ", label_file)
    label = {}
    with open(label_file, "r", encoding='utf-8') as infile:
        # load and check keys
        data = json.loads(infile.readline())
        if "tweet_city" in data:
            city_key, id_key, lat_key, lon_key = label_keys[0]
        else:
            city_key, id_key, lat_key, lon_key = label_keys[1]

        tweet_city = data[city_key]
        tweet_country = data[city_key].split("-")[-1]
        tweet_id = data[id_key]
        lat = data[lat_key]
        lon = data[lon_key]

        label[tweet_id] = {
            "tweet_id":tweet_id,
            "tweet_city":tweet_city,
            "tweet_country":tweet_country,
            "tweet_latitude":lat,
            "tweet_longitude":lon,
        }

        for i, line in enumerate(infile):
            if i % 100 == 0:
                print("\x1b[2K\rLoading Labels: {:>5}".format(i), end="")

            data = json.loads(line)
            tweet_city = data[city_key]
            tweet_country = data[city_key].split("-")[-1]
            tweet_id = data[id_key]
            lat = data[lat_key]
            lon = data[lon_key]

            label[tweet_id] = {
                "tweet_id":tweet_id,
                "tweet_city":tweet_city,
                "tweet_country":tweet_country,
                "tweet_latitude":lat,
                "tweet_longitude":lon,
            }
    print()

    print("loading data: ", data_file)
    total_data = []
    with open(data_file, "r", encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i % 100 == 0:
                print("\x1b[2K\rLoading Data: {:>5}".format(i), end="")
            
            if i == 50000:
                break

            data = json.loads(line)
            if "id_str" in data:
                tweet_id = data["id_str"]
            else:
                tweet_id = data["hashed_tweet_id"]
            tweet_label = label[tweet_id]
            data = {
                "tweet_id":tweet_id,
                "tweet_city":tweet_label["tweet_city"],
                "tweet_country":tweet_label["tweet_country"],
                "tweet_longitude":tweet_label["tweet_longitude"],
                "tweet_latitude":tweet_label["tweet_latitude"],
                "text":" ".join(tokenizeRawTweetText(data["text"])),
                "raw_text":data["text"],
                "created_at":data["created_at"],
                "lang":data.get("lang", ""),
                "user_time_zone":data["user"]["time_zone"],
                "user_lang":data["user"]["lang"],
                "user_name":"" if data["user"]["name"] is None else " ".join(tokenizeRawTweetText(data["user"]["name"])),
                "user_location":"" if data["user"]["location"] is None else " ".join(tokenizeRawTweetText(data["user"]["location"])),
                "user_description":"" if data["user"]["description"] is None else " ".join(tokenizeRawTweetText(data["user"]["description"])),
            }
            total_data.append(data)
    print()

    # save data
    total_data = pd.DataFrame(total_data) 
    total_data.to_parquet(arg.output_path)

if __name__ == "__main__":
    preprocessing()
