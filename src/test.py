import sys
sys.path.append("/home/appleternity/workspace/lab/graph/baseline/src")

# src
from util import Timer, build_embedding, get_batch_data, build_sparse_tensor, load_dictionary
from data import load_data, load_label

# current folder
from feature import *
import config as conf
from model import HAttention

# lib
import numpy as np
import tensorflow as tf
import time
import ujson as json
import csv

# TODO:
# (1) load testing data
# (2) preprocessing testing data (tokenize)

def test_main():
    # load data
    with Timer("loading label", 1):
        #test_label = load_label(conf.test_label, conf, lambda x:x.split("-")[-1])
        test_label = load_label(conf.test_label, conf)
        
    with Timer("loading data", 1):
        test_data = load_data(conf.test_data, feature_extractor, conf)
    
    # process data
    dictionary = load_dictionary(conf.word_dictionary_path)
    class_dictionary = load_dictionary(conf.class_dictionary_path)

    with Timer("turn2id", 1):
        turn2id(
            data_list=[test_data],
            name_list=["text"],
            dictionary=dictionary
        )

    with Timer("get_class", 1):
        turn_label2id(
            label_list=[test_label],
            dictionary=class_dictionary
        )
    
    print("num of classes = ", len(class_dictionary))
        
    with Timer("create_data", 1):
        # create training data
        test = create_data(
            data_list=[test_data],
            label_list=[test_label],
            conf=conf
        )[0]

    # update information to config
    conf.output_size = len(class_dictionary)
    conf.vocab_size = len(dictionary) - 1
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            # build model
            with Timer("building model", 2):
                with tf.device("/cpu:0"):
                    model = HAttention(conf)
                    model.build_model()
                    sess.run(tf.global_variables_initializer())
                    model.saver.restore(sess, os.path.join(conf.model_dir, "model_e{}".format(conf.testing_epoch)))

            name_list = ("text", "label")

            # evaluation
            results = []
            predicts = []
            accs = []
            for count, total_count, (x_text, y_label) in get_batch_data(test, name_list, 32, phase="test"):
                with tf.device("/cpu:0"):
                    acc, result, predict = sess.run([model.acc, model.result, model.predict], feed_dict={
                        model.input_text:x_text,
                        model.input_label:y_label,
                        model.input_dropout_rate:1
                    })
                    results.append(result)
                    predicts.append(predict)
                    accs.append(acc)

            results = np.hstack(results)
            predicts = np.hstack(predicts)
            acc = np.sum(results) / results.shape[0]
            print("Valid Acc = {:.6f}".format(acc))
            #print(accs)

            with open(os.path.join(conf.model_dir, "testing_e{}.json".format(conf.testing_epoch)), 'w', encoding='utf-8', newline="") as outfile:
                writer = csv.writer(outfile)
                o = np.hstack([
                    results.reshape(-1, 1), 
                    predicts.reshape(-1, 1), 
                    test["label"].reshape(-1, 1)
                ])
                #json.dump(o.tolist(), outfile, indent=4)
                writer.writerows(o.tolist())


if __name__ == "__main__":
    test_main()
