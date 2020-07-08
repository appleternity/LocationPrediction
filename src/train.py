# current folder
from util import Timer, build_embedding, get_batch_data, build_sparse_tensor, load_dictionary, saveh5, readh5
from data import load_data, load_label, mongo_load_data, load_city_map
from feature import *
import config as conf
from model import HAttention
from evaluation import compute_median

# lib
import numpy as np
import tensorflow as tf
import time
import ujson as json
import csv
import geopy.distance
from pymongo import MongoClient
import h5py

# TODO:
# (1) when using "keep_training", check available first.
# (2) check save model & batch normalization

def turn2dict(label):
    return {
        l["tweet_id"]:l for l in label 
    }

def train_main():
    if conf.process_data:
        # load data
        """
        with Timer("loading label", 1):
            if conf.city == "country":
                train_label = load_label(conf.train_label, conf, lambda x:x.split("-")[-1])
                valid_label = load_label(conf.valid_label, conf, lambda x:x.split("-")[-1])
                test_label = load_label(conf.test_label, conf, lambda x:x.split("-")[-1])
            else:
                train_label = load_label(conf.train_label, conf)
                valid_label = load_label(conf.valid_label, conf)
                test_label = load_label(conf.test_label, conf)

        with Timer("loading data", 1):
            train_data = load_data(conf.train_data, feature_extractor, conf)
            valid_data = load_data(conf.valid_data, feature_extractor, conf)
            test_data = load_data(conf.test_data, feature_extractor, conf)
        """

        # load mongo data
        with Timer("loading mongo data", 1):
            query = {
                "tweet_country":{"$in":["ph", "lk", "de"]}
            }
            #query = {}
            data_projection = {
                "_id":False, 
                "tweet_id":True, 
                "text":True, 
                "raw_text":True,
                "user_lang":True,
                "user_time_zone":True,
                "created_at":True
            }
            label_projection = {
                "_id":False, 
                "tweet_id":True, 
                "tweet_city":True, 
                "tweet_country":True,
                "tweet_longitude":True,
                "tweet_latitude":True
            }
            train_data = mongo_load_data("train", query, data_projection)
            valid_data = mongo_load_data("valid", query, data_projection)
            test_data = mongo_load_data("test", query, data_projection)

            train_label = turn2dict(mongo_load_data("train", query, label_projection))
            valid_label = turn2dict(mongo_load_data("valid", query, label_projection))
            test_label = turn2dict(mongo_load_data("test", query, label_projection))

            city_map = load_city_map() 

        # process data
        with Timer("getting_dictinoary", 1):
            dictionary, char_dictionary = get_dictionary(
                data_list=(train_data, valid_data),
                name_list=["text"],
                tokenizer=None, 
                minfreq=conf.minfreq,
                conf=conf
            )

        with Timer("turn2id", 1):
            turn2id(
                data_list=(train_data, valid_data, test_data),
                name_list=["text"],
                dictionary=dictionary,
                char_dictionary=char_dictionary
            )

        with Timer("get_class", 1):
            if conf.keep_training:
                class_dictionary, country_dictionary = load_dictionary(conf.class_dictionary_path)
            else:
                class_dictionary, country_dictionary = get_classes(train_data, train_label)
                
                # save dictionary
                with open(conf.class_dictionary_path, 'w', encoding='utf-8') as outfile:
                    json.dump([class_dictionary, country_dictionary], outfile, indent=4)

            turn_label2id(
                label_list=(train_label, valid_label, test_label),
                dictionary=class_dictionary,
                country_dictionary=country_dictionary,
                city_map=city_map
            )
        print("num of classes = ", len(class_dictionary))
       
        with Timer("meta data", 1):
            if conf.keep_training:
                lang_dictionary, timezone_dictionary = load_dictionary(conf.meta_dictionary_path)
            else:
                lang_dictionary, timezone_dictionary = get_meta_class(
                    data_list=(train_data, valid_data)
                )
                with open(conf.meta_dictionary_path, 'w', encoding='utf-8') as outfile:
                    json.dump([lang_dictionary, timezone_dictionary], outfile, indent=4)

            turn_meta2id(
                data_list=(train_data, valid_data, test_data),
                lang_dictionary=lang_dictionary,
                timezone_dictionary=timezone_dictionary
            )
        print("num of lang = ", len(lang_dictionary))
        print("num of timezone = ", len(timezone_dictionary))
            
        with Timer("create_data", 1):
            # create training data
            train, valid, test = create_data(
                data_list=(train_data, valid_data, test_data),
                label_list=(train_label, valid_label, test_label),
                conf=conf
            )
        # save data to .h5 file
        saveh5(train, conf.train_file)
        saveh5(valid, conf.valid_file)
        saveh5(test, conf.test_file)

    else:
        # read from .h5 file
        print("loading processed data directly")
        train = readh5(conf.train_file)
        valid = readh5(conf.valid_file)
        test = readh5(conf.test_file)
        class_dictionary, country_dictionary = load_dictionary(conf.class_dictionary_path)
        dictionary = load_dictionary(conf.word_dictionary_path)
        dictionary, char_dictionary = dictionary["dict"], dictionary["char_dict"]
        lang_dictionary, timezone_dictionary = load_dictionary(conf.meta_dictionary_path)
    
    # geo information
    class_mapping = {v:k for k, v in class_dictionary.items()}
    mongo = MongoClient("localhost")["twitter_location"]
    geo_mapping = {
        r["city"]:(r["lat_value"], r["lon_value"]) 
        for r in mongo["city_map"].find(
            {}, 
            projection={"_id":False, "city":True, "lat_value":True, "lon_value":True}
        )
    }

    # update information to config
    conf.output_size = len(class_dictionary)
    conf.country_output_size = len(country_dictionary)
    conf.vocab_size = len(dictionary) - 1
    conf.char_size = len(char_dictionary) - 1
    conf.time_size = 24*6
    conf.timezone_size = len(timezone_dictionary)
    conf.lang_size = len(lang_dictionary)

    # scale longitude & latitude
    train["longitude"] = train["longitude"]
    valid["longitude"] = valid["longitude"]
    test["longitude"] = test["longitude"]
    
    train["latitude"] = train["latitude"]
    valid["latitude"] = valid["latitude"]
    test["latitude"] = test["latitude"]
    
    with open(conf.history_path, 'w', encoding='utf-8') as outfile:
        pass

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            # build model
            with Timer("building model", 2):
                model = HAttention(conf)
                model.build_model()

            if not conf.keep_training:
                print("initialize variable")
                # initialization
                sess.run(tf.global_variables_initializer())
                print("finish initialize")
                conf.start_epoch = -1
            else:
                model.saver.restore(sess, os.path.join(conf.model_dir, "model_e{}".format(conf.start_epoch)))
            
            train_name_list = ("text", "char", "time", "timezone", "lang", "label", "country", "longitude", "latitude")
            test_name_list = ("text", "char", "time", "timezone", "lang", "label")
            total_loss = 0
            total_acc = 0
            for e in range(conf.start_epoch+1, conf.epochs):
                with Timer("Training epoch [{:>3}]".format(e), 2):
                    for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                            y_label, y_country, y_long, y_lat) in get_batch_data(
                                    train, train_name_list, conf.batch_size):
                        
                        _, loss, acc = sess.run([model.optim, model.loss, model.acc], feed_dict={
                            model.input_text:x_text,
                            model.input_char:x_char,
                            model.input_time:x_time,
                            model.input_timezone:x_timezone,
                            model.input_lang:x_lang,
                            model.input_label:y_label,
                            model.input_country_label:y_country,
                            model.input_long_label:y_long,
                            model.input_lat_label:y_lat,
                            model.input_dropout_rate:conf.dropout_rate,
                        })
                        total_loss = (total_loss * (count-1) + loss) / count
                        total_acc = (total_acc * (count-1) + acc) / count
                        
                        if count % 100 == 0:
                            print("\riter: {}/{} [{:.2f}%] acc={:.3f}, loss={:.3f}".format(
                                count, total_count, count/total_count*100, total_acc, total_loss
                            ), end="")
 
                    print()
                
                # evaluation
                results = []
                predicts = []
                for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                        y_label) in get_batch_data(valid, test_name_list, conf.batch_size, phase="test"):
                    result, predict = sess.run([model.result, model.predict], feed_dict={
                        model.input_text:x_text,
                        model.input_time:x_time,
                        model.input_timezone:x_timezone,
                        model.input_lang:x_lang,
                        model.input_char:x_char,
                        model.input_label:y_label,
                        model.input_dropout_rate:0
                    })
                    results.append(result)
                    predicts.append(predict)

                results = np.hstack(results)
                predicts = np.hstack(predicts)
                acc = np.sum(results) / results.shape[0]
                print("Valid Acc = {:.6f}".format(acc))
               
                with open(os.path.join(conf.model_dir, "predict_e{}.json".format(e)), 'w', encoding='utf-8') as outfile:
                    json.dump(predicts.tolist(), outfile, indent=4)
                
                # save model
                model.saver.save(sess=sess, save_path=os.path.join(conf.model_dir, "model_e{}".format(e)))
                history_info = {"epoch":e, "valid_acc":acc, "train_acc":total_acc, "train_loss":total_loss}
                
                # test
                results = []
                predicts = []
                w1s = []
                w2s = []
                cw31s = []
                cw32s = []
                cw41s = []
                cw42s = []
                cw51s = []
                cw52s = []
                for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                        y_label) in get_batch_data(test, test_name_list, conf.batch_size, phase="test"):
                    result, predict, w1, w2, cw31, cw32, cw41, cw42, cw51, cw52 = sess.run([
                        model.result, 
                        model.predict,
                        model.weight_1,
                        model.weight_2,
                        model.char_weights["3_1"],
                        model.char_weights["3_2"],
                        model.char_weights["4_1"],
                        model.char_weights["4_2"],
                        model.char_weights["5_1"],
                        model.char_weights["5_2"],
                    ], feed_dict={
                        model.input_text:x_text,
                        model.input_time:x_time,
                        model.input_timezone:x_timezone,
                        model.input_lang:x_lang,
                        model.input_char:x_char,
                        model.input_label:y_label,
                        model.input_dropout_rate:0
                    })
                    results.append(result)
                    predicts.append(predict)
                    w1s.append(w1)
                    w2s.append(w2)
                    cw31s.append(cw31)
                    cw32s.append(cw32)
                    cw41s.append(cw41)
                    cw42s.append(cw42)
                    cw51s.append(cw51)
                    cw52s.append(cw52)

                results = np.hstack(results)
                predicts = np.hstack(predicts)
                w1s = np.concatenate(w1s)
                w2s = np.concatenate(w2s)
                cw31s = np.concatenate(cw31s)
                cw32s = np.concatenate(cw32s)
                cw41s = np.concatenate(cw41s)
                cw42s = np.concatenate(cw42s)
                cw51s = np.concatenate(cw51s)
                cw52s = np.concatenate(cw52s)

                acc = np.sum(results) / results.shape[0]
                print("Test Acc = {:.6f}".format(acc))

                # distance
                distance_list = []
                for p, lat, lon in zip(predicts, test["latitude"], test["longitude"]):
                    if class_mapping[p] not in geo_mapping: continue
                    p_lat, p_lon = geo_mapping[class_mapping[p]]
                    d = geopy.distance.distance((p_lat, p_lon), (lat, lon)).miles
                    distance_list.append(d)
                
                try:
                    avg_d = sum(distance_list)/len(distance_list)
                    med_d = compute_median(distance_list)
                except Exception:
                    avg_d = 0
                    med_d = 0
                print("avg distance={}, med distance={}".format(avg_d, med_d))
                history_info["test_acc"] = acc
                history_info["avg_distance"] = avg_d
                history_info["med_distance"] = med_d

                with open(os.path.join(conf.model_dir, "test_e{}.csv".format(e)), 'w', encoding='utf-8', newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["avg", avg_d, "med", med_d])
                    o = np.hstack([
                        results.reshape(-1, 1), 
                        predicts.reshape(-1, 1), 
                        test["label"].reshape(-1, 1)
                    ])
                    writer.writerows(o.tolist())
               
                with h5py.File(os.path.join(conf.model_dir, "weights_e{}.h5".format(e)), 'w') as outfile:
                    outfile.create_dataset("w1s", data=w1s)
                    outfile.create_dataset("w2s", data=w2s)
                    outfile.create_dataset("cw31s", data=cw31s)
                    outfile.create_dataset("cw32s", data=cw32s)
                    outfile.create_dataset("cw71s", data=cw41s)
                    outfile.create_dataset("cw72s", data=cw42s)
                    outfile.create_dataset("cw51s", data=cw51s)
                    outfile.create_dataset("cw52s", data=cw52s)

                with open(conf.history_path, 'a', encoding='utf-8') as outfile:
                    outfile.write(json.dumps(history_info) + "\n")

if __name__ == "__main__":
    train_main()
