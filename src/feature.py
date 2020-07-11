import numpy as np
import ujson as json
import os
from dateutil.parser import parse as parse_time
from collections import Counter

def get_dictionary(data, minfreq=10):
    data["text"] = data["text"].apply(lambda x: x.lower().split(" "))
    count = Counter(token for sent in data["text"].tolist() for token in sent)
    
    data["raw_text"] = data["raw_text"].apply(lambda x: x.lower())
    char_count = Counter(char for sent in data["raw_text"].tolist() for char in sent)
    
    # sorted
    dictionary = {"<unk>":1, "<nan>":0}
    for key, freq in count.items():
        if freq >= minfreq:
            dictionary[key] = len(dictionary)
   
    char_dictionary = {"<unk>":1, "<nan>":0}
    for key, freq in char_count.items():
        if freq >= minfreq:
            char_dictionary[key] = len(char_dictionary)

    return dictionary, char_dictionary

def get_classes(data):
    dictionary = {"<unk>":0}
    country_dictionary = {"<unk>":0}

    for city in data["tweet_city"].unique():
        dictionary[city] = len(dictionary)

    for country in data["tweet_country"].unique():
        country_dictionary[country] = len(country_dictionary)

    return dictionary, country_dictionary

def get_meta_class(data):
    lang_dictionary = {"<unk>":0}
    timezone_dictionary = {"<unk>":0}

    for lang in data["user_lang"].unique():
        lang_dictionary[lang] = len(lang_dictionary)

    for timezone in data["user_time_zone"].unique():
        timezone_dictionary[timezone] = len(timezone_dictionary)

    return lang_dictionary, timezone_dictionary

def turn2id(data_list, dictionary, char_dictionary, max_len=30, max_char_len=140):
    unk_word = dictionary["<unk>"]
    unk_char = char_dictionary["<unk>"]
    for data in data_list:
        data["text"] = data["text"].apply(lambda sent: [dictionary.get(t, unk_word) for t in sent[:max_len]])
        data["char"] = data["raw_text"].apply(lambda sent: [char_dictionary.get(c, unk_char) for c in sent[:max_char_len]])

def turn_label2id(data_list, dictionary, country_dictionary):
    unk_city = dictionary["<unk>"]
    unk_country = dictionary["<unk>"]
    for i, data in enumerate(data_list):
        data["tweet_city"] = data["tweet_city"].apply(lambda city: dictionary.get(city, unk_city))
        data["tweet_country"] = data["tweet_country"].apply(lambda country: dictionary.get(country, unk_country))

            # TODO: check if this will happen
            #if i != 3: # train & valid
            #    label[_id] = (
            #        city, 
            #        country, 
            #        city_map.get(l["tweet_city"], l["tweet_longitude"]), 
            #        city_map.get(l["tweet_city"], l["tweet_latitude"])
            #    )
            #else:
            #    label[_id] = (
            #        city,
            #        country,
            #        l["tweet_longitude"],
            #        l["tweet_latitude"]
            #    )

def my_time_parsing(t):
    hour, minute, _ = t.split(" ")[3].split(":")
    return int(int(hour)*6 + int(minute)//10)

def turn_meta2id(data_list, lang_dictionary, timezone_dictionary):
    unk_lang = lang_dictionary["<unk>"]
    unk_timezone = timezone_dictionary["<unk>"]
    for data in data_list:
        data["user_lang"] = data["user_lang"].apply(lambda lang: lang_dictionary.get(lang, unk_lang))
        data["user_time_zone"] = data["user_time_zone"].apply(lambda timezone: timezone_dictionary.get(timezone, unk_timezone))
        data["created_at"] = data["created_at"].apply(my_time_parsing)

def create_data(data_list, max_len=30, max_char_len=140):
    result = []
    for data_count, data in enumerate(data_list, 1):
        label_array = np.zeros(len(data), dtype=np.int16) # 3000
        country_array = np.zeros(len(data), dtype=np.int16) # < 3000
        long_array = np.zeros((len(data), 1), dtype=np.float32) # real value
        lat_array = np.zeros((len(data), 1), dtype=np.float32) # real value
        text_array = np.zeros((len(data), max_len), dtype=np.int32) # vocab_size
        char_array = np.zeros((len(data), max_char_len), dtype=np.int32) # char_size
        time_array = np.zeros(len(data), dtype=np.int16) # 360
        timezone_array = np.zeros(len(data), dtype=np.int16) # 24
        lang_array = np.zeros(len(data), dtype=np.int16) # < 1000

        total_length = len(data)
        for count, tweet in data.iterrows():
            if count%100 == 0:
                print("\x1b[2K\rCreating Data Part {}: {} / {} [{:.2f}%]".format(data_count, count, total_length, 100.0*count/total_length), end="")
            
            # label
            label_array[count] = tweet["tweet_city"]
            country_array[count] = tweet["tweet_country"]
            long_array[count][0] = tweet["tweet_longitude"]
            lat_array[count][0] = tweet["tweet_latitude"]

            # text
            text = tweet["text"]
            text_len = len(text) if len(text) <= max_len else max_len
            if text_len != 0:
                text_array[count, -text_len:] = text[:text_len]

            # char
            chars = tweet["char"]
            char_len = len(chars) if len(chars) <= max_char_len else max_char_len
            if char_len != 0:
                char_array[count, -char_len:] = chars[:char_len]

            # meta
            time_array[count] = tweet["created_at"]
            timezone_array[count] = tweet["user_time_zone"]
            lang_array[count] = tweet["user_lang"]

        print()
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

