from pymongo import MongoClient
import ujson as json

def main():
    mongo = MongoClient("localhost")["twitter_location"]["result"]

    # all correct
    query = {"$where":"""
        this.tweet_city==this.proposed_1.city   && 
        this.tweet_city==this.deepgeo.city      &&
        this.tweet_city==this.cnn.city          &&
        this.tweet_city==this.proposed_2.city
    """}
    count = mongo.count(query)
    print("all correct", count)
    res = mongo.find(query, projection={"_id":False, "raw_text":True, "tweet_city":True, "tweet_country":True}) 
    res = [r for r in res]
    with open("analysis_output/all_correct.json", 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4)

    # only one token
    query = {"$where":"""
        this.tweet_city!=this.proposed_1.city   && 
        this.tweet_city!=this.deepgeo.city      &&
        this.tweet_city!=this.cnn.city          &&
        this.tweet_city!=this.proposed_2.city
    """}
    count = mongo.count(query)
    print("all wrong", count)
    res = mongo.find(query, projection={"_id":False}) 
    res = [{
        "raw_text":r["raw_text"],
        "tweet_city":r["tweet_city"],
        "tweet_country":r["tweet_country"],
        "proposed_1":r["proposed_1"]["city"],
        "proposed_2":r["proposed_2"]["city"],
        "deepgeo":r["deepgeo"]["city"],
        "cnn":r["cnn"]["city"],
        "mapping":r["proposed_2"]["mapping"],
    } for r in res]
    with open("analysis_output/all_wrong.json", 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4)

    # we good
    query = {"$where":"""
        this.tweet_city==this.proposed_1.city   && 
        this.tweet_city!=this.deepgeo.city      &&
        this.tweet_city!=this.cnn.city          &&
        this.tweet_city==this.proposed_2.city
    """}
    count = mongo.count(query)
    print("we good", count)
    res = mongo.find(query, projection={"_id":False}) 
    res = [{
        "raw_text":r["raw_text"],
        "tweet_city":r["tweet_city"],
        "tweet_country":r["tweet_country"],
        "proposed_1":r["proposed_1"]["city"],
        "proposed_2":r["proposed_2"]["city"],
        "deepgeo":r["deepgeo"]["city"],
        "cnn":r["cnn"]["city"],
        "mapping":r["proposed_2"]["mapping"],
    } for r in res]
    with open("analysis_output/we_good.json", 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4)
    
    # we bad
    query = {"$where":"""
        this.tweet_city!=this.proposed_1.city   && 
        this.tweet_city==this.deepgeo.city      &&
        this.tweet_city==this.cnn.city          &&
        this.tweet_city!=this.proposed_2.city
    """}
    count = mongo.count(query)
    print("we bad", count)
    res = mongo.find(query, projection={"_id":False}) 
    res = [{
        "raw_text":r["raw_text"],
        "tweet_city":r["tweet_city"],
        "tweet_country":r["tweet_country"],
        "proposed_1":r["proposed_1"]["city"],
        "proposed_2":r["proposed_2"]["city"],
        "deepgeo":r["deepgeo"]["city"],
        "cnn":r["cnn"]["city"],
        "mapping":r["proposed_2"]["mapping"],
    } for r in res]
    with open("analysis_output/we_bad.json", 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4)


if __name__ == "__main__":
    main()
