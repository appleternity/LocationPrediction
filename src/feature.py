import numpy as np
import ujson as json
import os.path
from dateutil.parser import parse as parse_time

def feature_extractor(tweet, id_key):
    d = {}
    d["text"] = tweet["text"]
    d["id"] = tweet[id_key]
    
    return d

def get_dictionary(data_list, name_list, tokenizer, minfreq, conf):
    # check if there is any existed dictionary
    if os.path.isfile(conf.word_dictionary_path):
        with open(conf.word_dictionary_path, 'r', encoding='utf-8') as infile:
            dictionary = json.load(infile)
            return dictionary["dict"], dictionary["char_dict"]
    
    count = {}
    char_count = {}
    for data in data_list:
        for tweet in data:
            for name in name_list:
                text = tweet[name].lower()
                tokens = text.split(" ")
                #tokens = tokenizer(text)
                #tokens = [t.lower() for t in tokens]
                #tweet[name] = tokens
                for token in tokens:
                    count[token] = count.get(token, 0) + 1

                raw_text = tweet["raw_text"].lower()
                for t in raw_text:
                    char_count[t] = char_count.get(t, 0) + 1

    # sorted
    dictionary = {"<unk>":1, "<nan>":0}
    for key, freq in count.items():
        if freq >= minfreq:
            dictionary[key] = len(dictionary)
   
    char_dictionary = {"<unk>":1, "<nan>":0}
    for key, freq in char_count.items():
        if freq >= minfreq:
            char_dictionary[key] = len(char_dictionary)

    with open(conf.word_dictionary_path, 'w', encoding='utf-8') as outfile:
        json.dump({"dict":dictionary, "char_dict":char_dictionary}, outfile, indent=4)

    return dictionary, char_dictionary

def get_classes(data, label):
    dictionary = {"<unk>":0}
    country_dictionary = {"<unk>":0}

    for tweet in data:
        c = label[tweet["tweet_id"]]["tweet_city"]
        if c not in dictionary:
            dictionary[c] = len(dictionary)

        country = label[tweet["tweet_id"]]["tweet_country"]
        if country not in country_dictionary:
            country_dictionary[country] = len(country_dictionary)

    return dictionary, country_dictionary

def get_meta_class(data_list):
    lang_dictionary = {"<unk>":0}
    timezone_dictionary = {"<unk>":0}
    for data in data_list:
        for t in data:
            if t["user_lang"] and t["user_lang"] not in lang_dictionary:
                lang_dictionary[t["user_lang"]] = len(lang_dictionary)
            if t["user_time_zone"] and t["user_time_zone"] not in timezone_dictionary:
                timezone_dictionary[t["user_time_zone"]] = len(timezone_dictionary)
    return lang_dictionary, timezone_dictionary

def turn_meta2id(data_list, lang_dictionary, timezone_dictionary):
    for data in data_list:
        for t in data:
            t["user_lang"] = lang_dictionary.get(t["user_lang"], 0)
            t["user_time_zone"] = timezone_dictionary.get(t["user_time_zone"], 0)
            d = parse_time(t["created_at"])
            t["created_at"] = int(d.hour*6 + int(d.minute/10))

def turn2id(data_list, name_list, dictionary, char_dictionary):
    for data in data_list:
        for tweet in data:
            for name in name_list:
                data = tweet[name].lower()
                tokens = data.split(" ")
                tokens = [dictionary.get(t, 1) for t in tokens[:30]]
                tweet[name] = tokens
                
                raw_text = tweet["raw_text"].lower()
                tokens = [char_dictionary.get(t, 1) for t in raw_text[:140]]
                tweet[name+"_char"] = tokens

def turn_label2id(label_list, dictionary, country_dictionary, city_map):
    for i, label in enumerate(label_list):
        for _id, l in label.items():
            #c = l.split("-")[-1]
            city = dictionary.get(l["tweet_city"], 0)
            country = country_dictionary.get(l["tweet_country"], 0)
            if i != 3: # train & valid
                label[_id] = (
                    city, 
                    country, 
                    city_map.get(l["tweet_city"], l["tweet_longitude"]), 
                    city_map.get(l["tweet_city"], l["tweet_latitude"])
                )
            else:
                label[_id] = (
                    city,
                    country,
                    l["tweet_longitude"],
                    l["tweet_latitude"]
                )

"""
def turn_category2id(data_list, name_list, dictionary):
    for data in data_list:
        for tweet in data:
            for name in name_list:
                tweet[name] = dictionary[name].get(tweet[name], 0)
"""

def create_data(data_list, label_list, conf):
    result = []
    for data, label in zip(data_list, label_list):
        label_array = np.zeros(len(data), dtype=np.int16)
        country_array = np.zeros(len(data), dtype=np.int16)
        long_array = np.zeros((len(data), 1), dtype=np.float32)
        lat_array = np.zeros((len(data), 1), dtype=np.float32)
        text_array = np.zeros((len(data), conf.text_len), dtype=np.int32)
        char_array = np.zeros((len(data), conf.char_len), dtype=np.int32)
        time_array = np.zeros(len(data), dtype=np.int16)
        timezone_array = np.zeros(len(data), dtype=np.int16)
        lang_array = np.zeros(len(data), dtype=np.int16)

        for count, tweet in enumerate(data):
            # label
            label_array[count] = label[tweet["tweet_id"]][0]
            country_array[count] = label[tweet["tweet_id"]][1]
            long_array[count][0] = label[tweet["tweet_id"]][2]
            lat_array[count][0] = label[tweet["tweet_id"]][3]

            # text
            text = tweet["text"]
            text_len = len(text) if len(text) <= conf.text_len else conf.text_len
            if text_len != 0:
                text_array[count, -text_len:] = text[:conf.text_len]

            # char
            chars = tweet["text_char"]
            char_len = len(chars) if len(chars) <= conf.char_len else conf.char_len
            if char_len != 0:
                char_array[count, -char_len:] = chars[:conf.char_len]

            # meta
            time_array[count] = tweet["created_at"]
            timezone_array[count] = tweet["user_time_zone"]
            lang_array[count] = tweet["user_lang"]

        result.append({
            "text":text_array,
            "label":label_array,
            "char":char_array,
            "country":country_array,
            "longitude":long_array,
            "latitude":lat_array,
            "lang":lang_array,
            "time":time_array,
            "timezone":timezone_array,
        })

    return result

