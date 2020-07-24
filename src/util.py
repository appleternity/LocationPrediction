import ujson as json
from datetime import datetime
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import h5py

class Timer(object):
    def __init__(self, name=None, verbose=2):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.t1 = datetime.now()
        print()
        if self.name:
            if self.verbose == 1:
                print("Starting {} ...".format(self.name))
            elif self.verbose == 2:
                print("Starting {} at {} ...".format(self.name, self.t1))
        else:
            if self.verbose == 1:
                print("Starting ...")
            elif self.verbose == 2:
                print("Starting at {} ...".format(self.t1))
        return self

    def __exit__(self, type, value, traceback):
        self.t2 = datetime.now()
        t = (self.t2 - self.t1).total_seconds()
        if self.name:
            if self.verbose == 1:
                print("Finishing {} in {} sec ...".format(self.name, t))
            elif self.verbose == 2:
                print("Finishing {} in {} sec at {} ...".format(self.name, t, self.t2))
        else:
            if self.verbose == 1:
                print("Finishing in {} sec ...".format(t))
            elif self.verbose == 2:
                print("Finishing in {} sec at {} ...".format(t, self.t2))

class Configuration(object):
    def __init__(self, arg=None, path=None):
        if arg is not None:
            # setting
            self.vocab_size     = None
            self.char_size      = None
            self.output_size    = None
            self.time_size      = None
            self.timezone_size  = None
            self.lang_size      = None
            self.max_len        = arg.max_len
            self.max_char_len   = arg.max_char_len
            self.minfreq        = arg.minfreq
           
            self.emb_dim        = arg.emb_dim
            self.hidden_dim     = arg.hidden_dim
            self.num_head       = arg.num_head
            self.layer_num      = arg.layer_num
            
            self.char_dim        = arg.char_dim
            self.char_hidden_dim = arg.char_hidden_dim
            self.char_num_head   = arg.char_num_head
            self.char_layer_num  = arg.char_layer_num
            self.filter_list     = arg.filter_list

            #self.use_meta = arg.use_meta
            #self.meta_dim = arg.meta_dim
            #self.use_coordinate       = arg.use_coordinate
            #self.normalize_coordinate = arg.normalize_coordinate

            # training setting
            self.dropout_rate   = arg.dropout_rate
            self.learning_rate  = arg.learning_rate
            self.batch_size     = arg.batch_size
            self.epochs         = arg.epochs
            self.reg            = arg.reg
            self.reg_weight     = arg.reg_weight

            self.train_data = arg.train_data

        elif path is not None:
            with open(path, 'r', encoding='utf-8') as infile:
                self.__dict__.update(json.load(infile))

                self.filter_list = {int(k):int(v) for k, v in self.filter_list.items()}
        else:
            print("Please at least pass arg or path to initialize the Configuration object.")

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(self.__dict__, outfile, indent=4)

def save_dictionary(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def load_dictionary(path):
    with open(path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        return data

def build_embedding(dictionary):
    word2vec = load_word2vec()
    wordlist = sorted(dictionary.items(), key = lambda x:x[1])
    wordvec = [
        word2vec[word] if word in word2vec else np.random.normal(size=300)
        for word in wordlist
    ]
    wordvec[0] = np.zeros(300)
    wordvec = np.vstack(wordvec)
    return wordvec.astype(np.float32)

def load_word2vec():
    model_path = "/apple_data/corpus/word2vec/GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(datapath(model_path), binary=True)  # C binary format
    return model

def test_timer():
    import time
    with Timer(name="Testing", verbose=2):
        time.sleep(5)

def get_batch_data(data, name_list, batch_size, phase="train"):
    if phase == "train":
        true_length = data[name_list[0]].shape[0]
        index_list = np.random.permutation(true_length)
        index_list = index_list[:true_length//batch_size*batch_size] # drop the rest of data
        index_list = index_list.reshape(-1, batch_size)

        total_count = index_list.shape[0]
        for count, indices in enumerate(index_list, 1):
            yield count, total_count, (data[name][indices] for name in name_list)

    elif phase == "test" or phase == "valid":
        true_length = data[name_list[0]].shape[0]
        total_count = int(np.ceil(true_length / batch_size))
        count = 0
        for i in range(0, true_length, batch_size):
            start = i
            end = min(i+batch_size, true_length)
            indices = [j for j in range(start, end)]
            count += 1
            yield count, total_count, (data[name][indices] for name in name_list)

def saveh5(data, filename):
    with h5py.File(filename, 'w') as outfile:
        for key, matrix in data.items():
            outfile.create_dataset(key, data=matrix)

def readh5(filename, verbose=True):
    print("Loading {}".format(filename))
    with h5py.File(filename, 'r') as infile:
        data = {}
        for key in infile.keys():
            matrix = np.empty(infile[key].shape, infile[key].dtype)
            infile[key].read_direct(matrix)
            data[key] = matrix
            print("{} Loaded with shape = {}, dtype={}".format(key, str(matrix.shape), matrix.dtype))
    print()
    return data

if __name__ == "__main__":
    test_timer()
        

