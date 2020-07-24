from util import *
from feature import *
from config import *
from model import HAttention

# lib
import numpy as np
import tensorflow as tf
import time
import ujson as json
import csv
import argparse

# TODO:
# (1) load testing data
# (2) preprocessing testing data (tokenize)

def parse_arg():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--model_folder", dest="model_folder", help="The path to the target model's folder", type=str, required=True)
    parser.add_argument("--target_epoch", dest="target_epoch", help="Specify the model for testing.", type=int, required=True)
    parser.add_argument("--gpu", dest="gpu", help="gpu setting", type=str, default="0")

    return parser.parse_args()

def test_main():
    arg = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

    folder_path = arg.model_folder
    test = readh5(os.path.join(folder_path, "test.h5"))
    dictionary = load_dictionary(os.path.join(folder_path, "dictionary.json"))
    char_dictionary = load_dictionary(os.path.join(folder_path, "char_dictionary.json"))
    class_dictionary = load_dictionary(os.path.join(folder_path, "class_dictionary.json"))
    country_dictionary = load_dictionary(os.path.join(folder_path, "country_dictionary.json"))
    config = Configuration(path=os.path.join(folder_path, "config.json"))

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

            name_list = ("text", "char", "label")

            # evaluation
            results = []
            predicts = []
            accs = []
            for count, total_count, (x_text, x_char, y_label) in get_batch_data(test, name_list, 256, phase="test"):
                with tf.device("/cpu:0"):
                    acc, result, predict = sess.run([model.acc, model.result, model.predict], feed_dict={
                        model.input_text:x_text,
                        model.input_char:x_char,
                        model.input_label:y_label,
                        model.input_dropout_rate:0.0
                    })
                    results.append(result)
                    predicts.append(predict)
                    accs.append(acc)

            results = np.hstack(results)
            predicts = np.hstack(predicts)
            acc = np.sum(results) / results.shape[0]
            print("Valid Acc = {:.6f}".format(acc))

            with open(os.path.join(folder_path, "testing_e{}.json".format(arg.target_epoch)), 'w', encoding='utf-8', newline="") as outfile:
                writer = csv.writer(outfile)
                o = np.hstack([
                    results.reshape(-1, 1), 
                    predicts.reshape(-1, 1), 
                    test["label"].reshape(-1, 1)
                ])
                writer.writerows(o.tolist())

if __name__ == "__main__":
    test_main()
