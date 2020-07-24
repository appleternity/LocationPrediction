from util import *
from feature import *
from config import *
from model import HAttention

# lib
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import ujson as json
import csv
import argparse
from twokenize import tokenizeRawTweetText 

# TODO:
# (1) load testing data
# (2) preprocessing testing data (tokenize)

def parse_arg():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--model_folder", dest="model_folder", help="The path to the target model's folder", type=str, required=True)
    parser.add_argument("--target_epoch", dest="target_epoch", help="Specify the model for testing.", type=int, required=True)
    parser.add_argument("--text_file", dest="text_file", help="The path to the testing text file.", type=str, required=True)
    parser.add_argument("--output_file", dest="output_file", help="The path to the output file.", type=str, required=True)
    parser.add_argument("--gpu", dest="gpu", help="gpu setting", type=str, default="0")

    return parser.parse_args()

def inference_main():
    arg = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

    folder_path = arg.model_folder
    dictionary = load_dictionary(os.path.join(folder_path, "dictionary.json"))
    char_dictionary = load_dictionary(os.path.join(folder_path, "char_dictionary.json"))
    class_dictionary = load_dictionary(os.path.join(folder_path, "class_dictionary.json"))
    country_dictionary = load_dictionary(os.path.join(folder_path, "country_dictionary.json"))
    config = Configuration(path=os.path.join(folder_path, "config.json"))

    class_decode_dictionary = {v:k for k, v in class_dictionary.items()}
    country_decode_dictionary = {v:k for k, v in country_dictionary.items()}

    # process data 
    raw_text = []
    text = []
    char = []
    unk_word = dictionary.get("<unk>")
    unk_char = char_dictionary.get("<unk>")
    with open(arg.text_file, "r", encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()

            # tokenize
            tokens = [t.lower() for t in tokenizeRawTweetText(line)]
            chars = line.lower()

            # turn to vector
            sent = [dictionary.get(t, unk_word) for t in tokens[:config.max_len]]
            sent_char = [char_dictionary.get(c, unk_char) for c in chars[:config.max_char_len]]

            raw_text.append(line)
            text.append(sent)
            char.append(sent_char)

    # padding + turn to matrix
    text_matrix = np.ones([len(text), config.max_len], dtype=np.int32) * dictionary["<nan>"]
    char_matrix = np.ones([len(char), config.max_char_len], dtype=np.int32) * dictionary["<nan>"]
    for i, (t, c) in enumerate(zip(text, char)):
        text_len = len(t) if len(t) <= config.max_len else config.max_len
        char_len = len(c) if len(c) <= config.max_char_len else config.max_char_len
        if text_len != 0:
            text_matrix[i, -text_len:] = t[:text_len]
        if char_len != 0:
            char_matrix[i, -char_len:] = c[:char_len]

    print("text_matrix.shape = ", text_matrix.shape)
    print("char_matrix.shape = ", char_matrix.shape)
    data = {
        "text":text_matrix,
        "char":char_matrix,
    }

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # build model
            with Timer("building model", 2):
                with tf.device("/gpu:0"):
                    model = HAttention(config)
                    model.build_model()
                    sess.run(tf.global_variables_initializer())
                    model.saver.restore(sess, os.path.join(folder_path, "model_e{}".format(arg.target_epoch)))

            name_list = ("text", "char")

            # evaluation
            city_list = []
            country_list = []
            for count, total_count, (x_text, x_char) in get_batch_data(data, name_list, 256, phase="test"):
                with tf.device("/cpu:0"):
                    y_city, y_country = sess.run([model.predict, model.country_predict], feed_dict={
                        model.input_text:x_text,
                        model.input_char:x_char,
                        model.input_dropout_rate:0.0
                    })
                    city_list.append(y_city)
                    country_list.append(y_country)

            city_list = np.hstack(city_list).reshape([-1, ])
            country_list = np.hstack(country_list).reshape([-1, ])
            index_list = [i for i in range(len(text))]
            
            # decode city & country
            city_list = [class_decode_dictionary[c] for c in city_list]
            country_list = [country_decode_dictionary[c] for c in country_list]

            result = {
                "text": raw_text,
                "city": city_list, 
                "country": country_list,
            }
            print(result)
            table = pd.DataFrame(result, index=index_list)
            table.to_csv(arg.output_file)

if __name__ == "__main__":
    inference_main()
