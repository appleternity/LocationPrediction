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

def get_batch_data(data, name_list, batch_size, label_name="label", phase="train"):
    if phase == "train":
        label = data[label_name]
        true_length = label.shape[0]
        length = true_length + batch_size - (true_length % batch_size)
        index_list = np.array([i for i in range(0, length)])
        index_list[index_list>=true_length] -= true_length
        np.random.shuffle(index_list)

        index_list = index_list.reshape(-1, batch_size)
        
        total_count = index_list.shape[0]
        for count, indices in enumerate(index_list, 1):
            yield count, total_count, (data[name][indices] for name in name_list)

    elif phase == "test" :
        label = data[label_name]
        true_length = label.shape[0]
        total_count = true_length // batch_size
        offset = 0 if true_length % batch_size == 0 else 1
        for i in range(0, (true_length//batch_size) + offset):
            start = i*batch_size
            end = i*batch_size+batch_size
            end = min(end, true_length)
            indices = [i for i in range(start, end)]
            yield i+1, total_count, (data[name][indices] for name in name_list)

def get_batch_data_class(data, name_list, batch_size, category, phase="train"):
    if phase == "train":
        label_index = data["category_dictionary"][category]
        true_length = label_index.shape[0]
        length = true_length + batch_size - (true_length % batch_size)
        if true_length < length:
            index_list = np.random.choice(label_index, length-true_length)
            label_index = np.hstack([label_index, index_list])
        np.random.shuffle(label_index)
        label_index = label_index.reshape(-1, batch_size)
        
        total_count = label_index.shape[0]
        for count, indices in enumerate(label_index, 1):
            yield count, total_count, (data[name][indices] for name in name_list)

    elif phase == "test":
        label = data["category_dictionary"][category]
        true_length = label.shape[0]
        total_count = true_length // batch_size
        for i in range(0, (true_length//batch_size) + 1):
            start = i*batch_size
            end = i*batch_size+batch_size
            end = min(end, true_length)
            indices = [i for i in range(start, end)]
            yield i+1, total_count, (data[name][indices] for name in name_list)

def get_batch_data_hierarchical(data, name_list, batch_size, phase="train"):
    if phase == "train":
        city_label = data["city_label"]
        true_length = city_label.shape[0]
        length = true_length + batch_size - (true_length % batch_size)
        index_list = np.array([i for i in range(0, length)])
        index_list[index_list>=true_length] -= true_length
        np.random.shuffle(index_list)

        index_list = index_list.reshape(-1, batch_size)
        
        total_count = index_list.shape[0]
        for count, indices in enumerate(index_list, 1):
            yield count, total_count, (data[name][indices] for name in name_list)

    elif phase == "test" :
        label = data["city_label"]
        true_length = label.shape[0]
        total_count = true_length // batch_size
        for i in range(0, (true_length//batch_size) + 1):
            start = i*batch_size
            end = i*batch_size+batch_size
            end = min(end, true_length)
            indices = [i for i in range(start, end)]
            yield i+1, total_count, (data[name][indices] for name in name_list)

def build_sparse_tensor(array, indices, ones, size):
    indices = indices[:len(array)]
    ones = ones[:len(array)]
    shape = np.array([len(array), size], dtype=np.int32)
    indices = np.vstack([indices, array]).transpose()
    
    return indices, array, shape

def test_word2vec():
    #vectors = load_word2vec()
    #print(vectors.wv["hello"])
    embedding = build_embedding({
        "<unk>":0,
        "<nan>":1,
        "hello":2,
        "zero":5,
        "src":3,
        "aasdfasdfafds":4
    })
    print(embedding)

def saveh5(data, filename):
    with h5py.File(filename, 'w') as outfile:
        for key, matrix in data.items():
            outfile.create_dataset(key, data=matrix)

def readh5(filename):
    with h5py.File(filename, 'r') as infile:
        data = {}
        for key in infile.keys():
            matrix = np.empty(infile[key].shape, infile[key].dtype)
            infile[key].read_direct(matrix)
            data[key] = matrix
    return data

if __name__ == "__main__":
    #test_timer()
        
    test_word2vec()



