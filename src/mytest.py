from model import HAttention
import config as conf
import tensorflow as tf
from pprint import pprint
import os, os.path
import ujson as json

def test_1():
    conf.vocab_size = 100
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            with tf.device("/cpu:0"):
                model = HAttention(conf)
                model.build_model()
                pprint(tf.global_variables())

def test_2():
    model_dir = "/apple_data/workspace/location/model/proposed_position_sum_country"
    word_dictionary_path = os.path.join(model_dir, "word_dictionary.json")
    test_data = "/apple_data/workspace/IE/data/WNUT/data/test_cnn.tweet.json"
    
    with open(word_dictionary_path, 'r', encoding='utf-8') as infile:
        dictionary = json.load(infile)

    outfile = open(test_data+".test", 'w', encoding='utf-8')
    with open(test_data, 'r', encoding='utf-8') as infile:
        for line in infile:
            tweet = json.loads(line)
            text = tweet["text"]
            tokens = text.split(" ")
            for t in tokens:
                i = dictionary.get(t, 1)
                outfile.write("{} ".format(i))
            outfile.write(" ||| "+text+"\n")


if __name__ == "__main__":
    test_2()
