from pymongo import MongoClient
import os, os.path
import ujson as json
import h5py
import csv
from dateutil.parser import parse as parse_time
from pprint import pprint

def build_result_mongo():
    model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_joint_weight"
    mongo = MongoClient("localhost")["twitter_location"]

    #for r in mongo["test"].find({}):
    #    mongo["result"].insert(r)

    with open(os.path.join(model_dir, "word_dictionary.json"), 'r', encoding='utf-8') as infile:
        dictionary = json.load(infile)
        word_dictionary = dictionary["dict"]
        char_dictionary = dictionary["char_dict"]

    for i, r in enumerate(mongo["result"].find({}), 1):
        if i % 10 == 0:
            print("\r{}".format(i), end="")
        text = r["text"].lower()
        tokens = text.split(" ")
        tokens = [str(word_dictionary.get(t, 1)) for t in tokens[:30]]
        token = "_".join(tokens)
       
        raw_text = r["raw_text"].lower()
        char_tokens = [str(char_dictionary.get(t, 1)) for t in raw_text[:140]]
        char_token = "_".join(char_tokens)

        d = parse_time(r["created_at"])
        time_value = int(d.hour*6 + int(d.minute/10))
        
        mongo["result"].update(
            {"_id":r["_id"]},
            {"$set":{"hash_value":token, "time_value":time_value, "char_hash_value":char_token}}
        )

def put_weight():
    model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_joint_weight"
    e = 2

    mongo = MongoClient("localhost")["twitter_location"]

    with h5py.File(os.path.join(model_dir, "test.h5"), 'r') as infile:
        text = infile["text"][:]
        label = infile["label"][:]
        time = infile["time"][:]
        char = infile["char"][:]

    with open(os.path.join(model_dir, "class_dictionary.json"), 'r', encoding='utf-8') as infile:
        class_dictionary = json.load(infile)[0]
        reverse_class_dictionary = {v:k for k, v in class_dictionary.items()}

    with open(os.path.join(model_dir, "test_e{}.csv".format(e)), 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        result = [row for row in reader]
        result = result[1:]

    for i, (t, r, l, ti, ch) in enumerate(zip(text, result, label, time, char)):
        if i % 10 == 0:
            print("\r{}".format(i), end="")

        t = t.tolist()
        t = [str(tt) for tt in t if tt != 0]
        h = "_".join(t)
        pred = int(r[1])
        tr = int(r[2])
        if tr != l:
            print("GG")
            quit()

        ch = ch.tolist()
        ch = [str(tt) for tt in ch if tt != 0]
        ch_h = "_".join(ch)

        true_label = reverse_class_dictionary[tr]

        # find in mongo
        res = None
        if true_label == "<unk>":
            query = {"hash_value":h, "time_value":int(ti), "char_hash_value":ch_h}
        else:
            query = {"hash_value":h, "time_value":int(ti), "char_hash_value":ch_h, "tweet_city":true_label}
        res = mongo["result"].find_one(query)
        if res is None:
            pprint(query)
            print("res is None")
            quit()

        c = class_dictionary.get(res["tweet_city"], 0)
        if c != tr:
            print("c != tr   {} != {}".format(c, tr))
            #print(t)
            #print(res)
            continue

        mongo["result"].update({"_id":res["_id"]}, {
            "$set":{
                "proposed_2":{
                    "city": reverse_class_dictionary[int(pred)],
                    "city_index": int(pred),
                    "mapping":i
                }
            }
        })

def proposed_1():
    model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_subword"
    filename = os.path.join(model_dir, "test_e2.csv")
    mongo = MongoClient("localhost")["twitter_location"]
    
    with open(os.path.join(model_dir, "class_dictionary.json"), 'r', encoding='utf-8') as infile:
        class_dictionary = json.load(infile)[0]
        reverse_class_dictionary = {v:k for k, v in class_dictionary.items()}
    
    answer = [
        r
        for r in mongo["test"].find(
            {},
            projection={"tweet_latitude":True, "tweet_longitude":True, "tweet_city":True}
        )
    ]

    with open(filename, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for i, (row, ans) in enumerate(zip(reader, answer)):
            true_city = reverse_class_dictionary[int(row[2])]
            pred_city = reverse_class_dictionary[int(row[1])]
        
            if true_city != ans["tweet_city"]:
                print(true_city)

            mongo["result"].update({"_id":ans["_id"]}, {
                "$set": {
                    "proposed_1": {
                        "city":pred_city,
                        "city_index":int(row[1]),
                        "mapping":i
                    }
                }
            })

def deepgeo():
    model_dir = "../twitter-deepgeo/result"
    filename = os.path.join(model_dir, "result_textonly.json")
    mongo = MongoClient("localhost")["twitter_location"]
    
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    with open("../twitter-deepgeo/data/test/label.tweet.json", 'r', encoding='utf-8') as infile:
        labels = []
        for line in infile:
            d = json.loads(line)
            labels.append(d)
    
    for row, label in zip(data, labels):
        if row[1] != label["city"]:
            print(row[1], label["city"])
        c_true = row[1]
        c_pred = row[0]
    
        mongo["result"].update({"tweet_id":label["hashed_tweet_id"]}, {
            "$set":{
                "deepgeo": {
                    "city":c_pred    
                }
            }
        })
        
def cnn():
    model_dir = "/apple_data/workspace/location/model/lab_cnn_pure_textcity"
    filename = os.path.join(model_dir, "predict_e18.json")
    mongo = MongoClient("localhost")["twitter_location"]
    
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        pred_list = data["predict"]
        true_list = data["y_labels"]
    
    with open("../twitter-deepgeo/data/test/label.tweet.json", 'r', encoding='utf-8') as infile:
        labels = []
        for line in infile:
            d = json.loads(line)
            labels.append(d)
    
    with open(os.path.join(model_dir, "class_dictionary"), 'r', encoding='utf-8') as infile:
        class_dictionary = json.load(infile)
        reverse_class_dictionary = {v:k for k, v in class_dictionary.items()}

    for c_pred, c_true, label in zip(pred_list, true_list, labels):
        c_true = reverse_class_dictionary[c_true]
        c_pred = reverse_class_dictionary[c_pred]
        if c_true != label["city"]:
            print(c_true, label["city"])
    
        mongo["result"].update({"tweet_id":label["hashed_tweet_id"]}, {
            "$set":{
                "cnn": {
                    "city":c_pred    
                }
            }
        })

def main():
    #cnn()
    #deepgeo()
    #proposed_1()
    put_weight()
    #build_result_mongo()

if __name__ == "__main__":
    main()

