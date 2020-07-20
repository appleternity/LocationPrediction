# current folder
from util import *
from data import load_data, load_city_map
from feature import *
from config import *

import numpy as np

def test_data():
    note = "1m"
    folder_path = os.path.join(model_dir, "location_{}".format(note))

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

    # decode dictionary
    char_decode = {i:c for c, i in char_dictionary.items()}
    text_decode = {i:t for t, i in dictionary.items()}

    # reconstruct train
    sample_size = 100
    index = np.random.permutation(train["text"].shape[0])[:10]
    text = train["text"][index]
    char = train["char"][index]
    for text_sample, char_sample in zip(text, char):
        raw_text = " ".join([text_decode[t] for t in text_sample])
        raw_char = "".join([char_decode[c] for c in char_sample])
        print("=============Train==============")
        print(raw_text)
        print(text_sample)
        print(raw_char)
        print(char_sample)
        print()

    # reconstruct valid
    sample_size = 100
    index = np.random.permutation(valid["text"].shape[0])[:10]
    text = valid["text"][index]
    char = valid["char"][index]
    for text_sample, char_sample in zip(text, char):
        raw_text = " ".join([text_decode[t] for t in text_sample])
        raw_char = "".join([char_decode[c] for c in char_sample])
        print("=============Valid==============")
        print(raw_text)
        print(text_sample)
        print(raw_char)
        print(char_sample)
        print()

    # reconstruct test
    sample_size = 100
    index = np.random.permutation(test["text"].shape[0])[:10]
    text = test["text"][index]
    char = test["char"][index]
    for text_sample, char_sample in zip(text, char):
        raw_text = " ".join([text_decode[t] for t in text_sample])
        raw_char = "".join([char_decode[c] for c in char_sample])
        print("=============Test==============")
        print(raw_text)
        print(text_sample)
        print(raw_char)
        print(char_sample)
        print()



def main():
    test_data()

if __name__ == "__main__":
    main()
