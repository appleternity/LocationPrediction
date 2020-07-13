# current folder
from util import *
from data import load_data, load_city_map
from feature import *
from config import *
from model import HAttention
from evaluation import compute_median

# lib
import numpy as np
import tensorflow as tf
import time
import ujson as json
import csv
import geopy.distance
import h5py
import argparse

# TODO:
# (1) when using "keep_training", check available first.
# (2) check save model & batch normalization

def str2dict(s):
    filter_dict = {}
    for info in s.split("-"):
        kernel_size, num = info.split(":")
        filter_dict[int(kernel_size)] = int(num)
    return filter_dict

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arg():
    parser = argparse.ArgumentParser(description="Train the model.")

    # model setting
    #parser.add_argument("--vocab_size", dest="vocab_size", help="vocabulary size of the model", type=int, default=None)
    #parser.add_argument("--char_size", dest="char_size", help="character size of the model", type=int, default=None) 
    #parser.add_argument("--output_size", dest="output_size", help="output size of the model", type=int, default=None)

    parser.add_argument("--max_len", dest="max_len", help="maximum length of the tokens", type=int, default=30)
    parser.add_argument("--max_char_len", dest="max_char_len", help="maximum length of the characters", type=int, default=140)
    parser.add_argument("--minfreq", dest="minfreq", help="minimum frequency of the vocabulary and character", type=int, default=10)

    parser.add_argument("--emb_dim", dest="emb_dim", help="word embedding dimension", type=int, default=200)
    parser.add_argument("--hidden_dim", dest="hidden_dim", help="hidden dimension", type=int, default=200)
    parser.add_argument("--num_head", dest="num_head", help="number of head of the transformer", type=int, default=10)

    parser.add_argument("--char_dim", dest="char_dim", help="character embedding dimension", type=int, default=100)
    parser.add_argument("--char_hidden_dim", dest="char_hidden_dim", help="character hidden dimension", type=int, default=100)
    parser.add_argument("--char_num_head", dest="char_num_head", help="number of head of the character transformer", type=int, default=8)
    parser.add_argument("--filter", dest="filter_list", help="filter configuration of the character CNN, ex: 3:64-4:64", type=str2dict, default={3:64, 4:64, 5:64, 6:64, 7:64})

    parser.add_argument("--use_meta", dest="use_meta", help="whether use meta feature or not", type=str2bool, default=False)
    parser.add_argument("--meta_dim", dest="meta_dim", help="dimension of the meta data", type=int, default=50)
    
    parser.add_argument("--use_coordinate", dest="use_coordinate", help="whether use coordinate feature or not", type=str2bool, default=False)
    parser.add_argument("--normalize_coordinate", dest="normalize_coordinate", help="whether normalize the coordinate or not", type=str2bool, default=False)

    # training setting
    parser.add_argument("--dropout_rate", dest="dropout_rate", help="dropout rate across the model", type=float, default=0.05)
    parser.add_argument("--learning_rate", dest="learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", dest="batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--epochs", dest="epochs", help="number of epochs for training", type=int, default=30)
    parser.add_argument("--reg", dest="reg", help="whether use regularizer or not", type=str2bool, default=False)
    parser.add_argument("--reg_weight", dest="reg_weight", help="weighting for regularizer", type=float, default=1e-4)

    # other setting
    parser.add_argument("--data_redo", dest="data_redo", help="re-process the data again", type=str2bool, default=False)
    parser.add_argument("--note", dest="note", help="note for the model name", type=str, default="")
    parser.add_argument("--gpu", dest="gpu", help="gpu setting", type=str, default="0")
    parser.add_argument("--train_data", dest="train_data", help="training_data", type=str, default="train")

    #geo_cord = False
    #meta_feature = False

    return parser.parse_args()

def check_data_exist(folder_path):
    return all(os.path.isfile(os.path.join(folder_path, "{}.h5".format(data))) for data in ["train", "valid", "test"])

def train_main():
    arg = parse_arg()
    print(arg)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    config = Configuration(arg=arg)

    folder_path = os.path.join(model_dir, "location_{}".format(arg.note))
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    if not check_data_exist(folder_path) or arg.data_redo:
        with Timer("loading data", 1):
            train = load_data(phrase=config.train_data)
            valid = load_data(phrase="valid")
            test = load_data(phrase="test")
            city_map = load_city_map() 

        with Timer("getting dictinoary", 1):
            dictionary, char_dictionary = get_dictionary(
                data=train,
                minfreq=config.minfreq,
            )
            print("num of words = ", len(dictionary))
            print("num of chars = ", len(char_dictionary))

        with Timer("turning text to id", 1):
            turn2id(
                data_list=(train, valid, test),
                dictionary=dictionary,
                char_dictionary=char_dictionary
            )

        with Timer("getting class dictionary", 1):
            class_dictionary, country_dictionary = get_classes(train)
            print("num of cities = ", len(class_dictionary))
            print("num of countries = ", len(country_dictionary))

        with Timer("turning label to id", 1):
            turn_label2id(
                data_list=(train, valid, test),
                dictionary=class_dictionary,
                country_dictionary=country_dictionary,
            )
       
        with Timer("getting meta dictionary", 1):
            lang_dictionary, timezone_dictionary = get_meta_class(train)
            print("num of languages = ", len(lang_dictionary))
            print("num of timezones = ", len(timezone_dictionary))

        with Timer("turning meta to id", 1):
            turn_meta2id(
                data_list=(train, valid, test),
                lang_dictionary=lang_dictionary,
                timezone_dictionary=timezone_dictionary
            )
        print("num of lang = ", len(lang_dictionary))
        print("num of timezone = ", len(timezone_dictionary))
            
        with Timer("create_data", 1):
            # create training data
            train, valid, test = create_data(
                data_list=(train, valid, test), 
                max_len=config.max_len, 
                max_char_len=config.max_char_len,
            )

        # save all dictionary
        save_dictionary(dictionary, os.path.join(folder_path, "dictionary.json"))
        save_dictionary(char_dictionary, os.path.join(folder_path, "char_dictionary.json"))
        save_dictionary(class_dictionary, os.path.join(folder_path, "class_dictionary.json"))
        save_dictionary(country_dictionary, os.path.join(folder_path, "country_dictionary.json"))
        save_dictionary(lang_dictionary, os.path.join(folder_path, "lang_dictionary.json"))
        save_dictionary(timezone_dictionary, os.path.join(folder_path, "timezone_dictionary.json"))
        
        # save data to .h5 file
        saveh5(train, os.path.join(folder_path, "train.h5"))
        saveh5(valid, os.path.join(folder_path, "valid.h5"))
        saveh5(test, os.path.join(folder_path, "test.h5"))

    else:
        # read from .h5 file
        print("loading processed data directly")
        train = readh5(os.path.join(folder_path, "train.h5"))
        valid = readh5(os.path.join(folder_path, "valid.h5"))
        test = readh5(os.path.join(folder_path, "test.h5"))
        dictionary = load_dictionary(os.path.join(folder_path, "dictionary.json"))
        char_dictionary = load_dictionary(os.path.join(folder_path, "char_dictionary.json"))
        class_dictionary = load_dictionary(os.path.join(folder_path, "class_dictionary.json"))
        country_dictionary = load_dictionary(os.path.join(folder_path, "country_dictionary.json"))
        lang_dictionary = load_dictionary(os.path.join(folder_path, "lang_dictionary.json"))
        timezone_dictionary = load_dictionary(os.path.join(folder_path, "timezone_dictionary.json"))
    
    # geo information
    class_mapping = {v:k for k, v in class_dictionary.items()}
    geo_mapping = load_city_map()

    # update information to config
    config.output_size = len(class_dictionary)
    config.country_output_size = len(country_dictionary)
    config.vocab_size = len(dictionary) - 1
    config.char_size = len(char_dictionary) - 1
    config.time_size = 24*6
    config.timezone_size = len(timezone_dictionary)
    config.lang_size = len(lang_dictionary)
    
    config.save(os.path.join(folder_path, "config.json"))

    # scale longitude & latitude
    if config.normalize_coordinate:
        train["longitude"] = train["longitude"] / 180.0
        valid["longitude"] = valid["longitude"] / 180.0
        test["longitude"] = test["longitude"] / 180.0
    
        train["latitude"] = train["latitude"] / 90.0
        valid["latitude"] = valid["latitude"] / 90.0
        test["latitude"] = test["latitude"] / 90.0
    
    with open(os.path.join(folder_path, "history.json"), 'w', encoding='utf-8') as outfile:
        pass

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

        sess = tf.Session(config=session_conf)
        with sess.as_default():

            # build model
            with Timer("building model", 2):
                model = HAttention(config)
                model.build_model()
            
            print("initialize variable")
            sess.run(tf.global_variables_initializer())

            """
            if not config.keep_training:
                print("initialize variable")
                # initialization
                sess.run(tf.global_variables_initializer())
                print("finish initialize")
                config.start_epoch = -1
            else:
                model.saver.restore(sess, os.path.join(config.model_dir, "model_e{}".format(config.start_epoch)))
            """

            train_name_list = ("text", "char", "time", "timezone", "lang", "label", "country", "longitude", "latitude")
            test_name_list = ("text", "char", "time", "timezone", "lang", "label")
            total_loss = 0
            total_acc = 0
            for e in range(0, config.epochs):
                with Timer("Training epoch [{:>3}]".format(e), 2):
                    for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                            y_label, y_country, y_long, y_lat) in get_batch_data(
                                    train, train_name_list, config.batch_size):
                        
                        _, loss, acc = sess.run([model.optim, model.loss, model.acc], feed_dict={
                            model.input_text:x_text,
                            model.input_char:x_char,
                            model.input_label:y_label,
                            model.input_country_label:y_country,
                            model.input_dropout_rate:config.dropout_rate,
                            #model.input_timezone:x_timezone,
                            #model.input_time:x_time,
                            #model.input_lang:x_lang,
                            #model.input_long_label:y_long,
                            #model.input_lat_label:y_lat,
                        })
                        total_loss = (total_loss * (count-1) + loss) / count
                        total_acc = (total_acc * (count-1) + acc) / count
                        
                        if count % 1 == 0:
                            print("\riter: {}/{} [{:.2f}%] acc={:.5f}, loss={:.5f}".format(
                                count, total_count, count/total_count*100, total_acc, total_loss
                            ), end="")

                    print()
                
                # evaluation
                results = []
                predicts = []
                for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                        y_label) in get_batch_data(valid, test_name_list, config.batch_size, phase="valid"):
                    result, predict, acc = sess.run([model.result, model.predict, model.acc], feed_dict={
                        model.input_text:x_text,
                        model.input_char:x_char,
                        model.input_label:y_label,
                        model.input_dropout_rate:0.0
                        #model.input_time:x_time,
                        #model.input_timezone:x_timezone,
                        #model.input_lang:x_lang,
                    })
                    results.append(result)
                    predicts.append(predict)
                    print(acc)

                results = np.hstack(results)
                predicts = np.hstack(predicts)
                acc = np.sum(results) / results.shape[0]
                print("Valid Acc = {:.6f}".format(acc))
               
                with open(os.path.join(folder_path, "predict_e{}.json".format(e)), 'w', encoding='utf-8') as outfile:
                    json.dump(predicts.tolist(), outfile, indent=4)
                
                # save model
                model.saver.save(sess=sess, save_path=os.path.join(folder_path, "model_e{}".format(e)))
                history_info = {"epoch":e, "valid_acc":acc, "train_acc":total_acc, "train_loss":total_loss}
                
                # test
                results = []
                predicts = []
                for count, total_count, (x_text, x_char, x_time, x_timezone, x_lang, 
                        y_label) in get_batch_data(test, test_name_list, config.batch_size, phase="test"):
                    result, predict, acc = sess.run([
                        model.result, 
                        model.predict,
                        model.acc,
                    ], feed_dict={
                        model.input_text:x_text,
                        model.input_char:x_char,
                        model.input_label:y_label,
                        model.input_dropout_rate:0.0
                        #model.input_time:x_time,
                        #model.input_timezone:x_timezone,
                        #model.input_lang:x_lang,
                    })
                    results.append(result)
                    predicts.append(predict)
                    print(acc)

                results = np.hstack(results)
                predicts = np.hstack(predicts)
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

                with open(os.path.join(folder_path, "test_e{}.csv".format(e)), 'w', encoding='utf-8', newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["avg", avg_d, "med", med_d])
                    o = np.hstack([
                        results.reshape(-1, 1), 
                        predicts.reshape(-1, 1), 
                        test["label"].reshape(-1, 1)
                    ])
                    writer.writerows(o.tolist())

                with open(os.path.join(folder_path, "history.json"), 'a', encoding='utf-8') as outfile:
                    outfile.write(json.dumps(history_info) + "\n")

if __name__ == "__main__":
    train_main()
