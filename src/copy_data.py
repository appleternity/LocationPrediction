import os
from config import *
import argparse 

def parse_arg():
    parser = argparse.ArgumentParser(description="Copy data to skip the building data part.")

    parser.add_argument("--target", dest="target", help="target folder name", required=True, type=str)
    parser.add_argument("--source", dest="source", help="source folder name", type=str, default="location_all_final_l2_d0.3")

    return parser.parse_args()

def copy_data():
    arg = parse_arg()

    # create folder
    try:
        os.mkdir(os.path.join(model_dir, arg.target))
    except:
        print("{} exists".format(arg.target))

    # data
    for data in ["train.h5", "valid.h5", "test.h5"]:
        command = "ln -s {} {}".format(
            os.path.join(model_dir, arg.source, data),
            os.path.join(model_dir, arg.target, data)
        )
        print(command)
        os.system(command)

    # dictionary
    for dictionary in [
        "char_dictionary.json", 
        "class_dictionary.json", 
        "country_dictionary.json",
        "dictionary.json",
        "lang_dictionary.json",
        "timezone_dictionary.json",
    ]:
        command = "cp {} {}".format(
            os.path.join(model_dir, arg.source, dictionary),
            os.path.join(model_dir, arg.target, dictionary),
        )
        print(command)
        os.system(command)

if __name__ == "__main__":
    copy_data()
