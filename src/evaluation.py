import requests
from bs4 import BeautifulSoup 
from pprint import pprint
import re
from pymongo import MongoClient
import csv
import geopy.distance
import os, os.path
import json
import h5py

pattern = re.compile("(?P<direction>[NEWS]) (?P<x1>\d+)° (?P<x2>\d+)' (?P<x3>\d+)''")
def change_unit(s):
    #print(s)
    res = pattern.findall(s)[0]
    sign = 1 if res[0] == "N" or res[0] == "E" else -1
    return sign * (int(res[1]) + int(res[2])/60 + int(res[3])/3600)

def get_location(loc):
    url = "http://www.geonames.org/search.html?q={}&country=".format(loc)
    res = requests.get(url)
    #print(res)
    html = BeautifulSoup(res.text, "html.parser")
    table = html.find("table", class_="restable")
    lat, lon = table.find_all("tr")[2].find_all("td")[-2:]

    # latitude
    lat = lat.text
    lat_value = change_unit(lat)
    #print(lat_value)

    # longitude
    lon = lon.text
    lon_value = change_unit(lon)
    #print(lon_value)

    return (lon, lat), (lon_value, lat_value)

def main():
    result = get_location("milton keynes")
    print(result)

def get_all_location():
    mongo = MongoClient("localhost")["twitter_location"]
    #city = set(r for r in mongo["train"].distinct("tweet_city"))
    #city.update(r for r in mongo["test"].distinct("tweet_city"))
    #city.update(r for r in mongo["valid"].distinct("valid_city"))
    city = [r for r in mongo["city_map"].find()]
    
    print(len(city))
    total = len(city)
    for i, c in enumerate(city):
        print("\r{:>5} / {}".format(i, total), end="   ")
        try:
            (lon, lat), (lon_value, lat_value) = get_location(c["city"].split("-")[0])
        except Exception:
            (lon, lat), (lon_value, lat_value) = (None, None), (None, None)

        mongo["city_map"].update(
            {"_id":c["_id"]},
            {"$set":{"lon":lon, "lat":lat, "lon_value":lon_value, "lat_value":lat_value}}
        )

def compute_median(l):
    l = sorted(l)
    if len(l) % 2 == 0:
        i = int(len(l)/2)
        return (l[i] + l[i-1])/2
    else:
        return l[int(len(l)/2)]

def evaluation():
    #model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_subword"
    model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_joint_weight3"
    filename = os.path.join(model_dir, "test_e2.csv")
    dictionary_name = os.path.join(model_dir, "class_dictionary.json")
    
    with open(dictionary_name, 'r', encoding='utf-8') as infile:
        class_dictionary = json.load(infile)[0]
        class_dictionary = {
            v:k
            for k, v in class_dictionary.items()
        }

    mongo = MongoClient("localhost")["twitter_location"]
    city = {
        r["city"]:(r["lat_value"], r["lon_value"]) 
        for r in mongo["city_map"].find(
            {}, 
            projection={"_id":False, "city":True, "lat_value":True, "lon_value":True}
        )
    }
    answer = [
        (r["tweet_latitude"], r["tweet_longitude"], r["tweet_city"])
        for r in mongo["test"].find(
            {},
            projection={"_id":False, "tweet_latitude":True, "tweet_longitude":True, "tweet_city":True}
        )
    ]

    error_distance_list = []
    city_acc_list = []
    country_acc_list = []
    with open(filename, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for i, (row, ans) in enumerate(zip(reader, answer)):
            if i == 0: continue
            if class_dictionary[int(row[2])] == "<unk>": continue
            c_true = city[class_dictionary[int(row[2])]]
            #if class_dictionary[int(row[2])] != ans[2]:
            #    print("???")
            #c_true = ans[0:2]
            c_pred = city[class_dictionary[int(row[1])]]
            error_distance = geopy.distance.distance(c_true, c_pred).miles
            error_distance_list.append(error_distance)
           
            true_city = class_dictionary[int(row[2])]
            pred_city = class_dictionary[int(row[1])]
            if true_city == pred_city:
                city_acc_list.append(1)
            else:
                city_acc_list.append(0)

            if true_city.split("-")[-1] == pred_city.split("-")[-1]:
                country_acc_list.append(1)
            else:
                country_acc_list.append(0)

    print("len = {}".format(len(error_distance_list)))
    print("error_distance avg = {}, error_distance median = {}".format(
        sum(error_distance_list)/len(error_distance_list),
        compute_median(error_distance_list) 
    ))

    print("city_acc = {}, country_acc = {}".format(
        sum(city_acc_list)/len(city_acc_list),
        sum(country_acc_list)/len(country_acc_list)
    ))

def compute_valid():
    model_dir = "/apple_data/workspace/location/model/proposed_nonreg_join_position_sum_city_joint3"
    
    with h5py.File(os.path.join(model_dir, "valid.h5"), 'r') as infile:
        label = infile["label"][:]

    for i in range(0, 10):
        filename = "predict_e{}.json".format(i)
        with open(os.path.join(model_dir, filename), 'r') as infile:
            predict = json.load(infile)
        
        result = [1 if l==p else 0 for l, p in zip(label, predict)]
        print("{} = {}".format(filename, sum(result)/len(result)))

if __name__ == "__main__":
    #main()
    #get_all_location()
    evaluation()
    #compute_valid()


