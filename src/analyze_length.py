from config import *
import pandas as pd
import numpy as np
import os

def analyze_length(phrase): 
    print("\nAnalyzing {} data".format(phrase))
    data = pd.read_parquet(os.path.join(data_dir, "{}.parquet".format(phrase)))

    # token
    data["text_token"] = data["text"].apply(lambda x: x.split(" "))
    data["token_length"] = data["text_token"].apply(lambda x: len(x))
    token_length = data["token_length"].to_numpy()
    for p in [50, 75, 80, 90, 95, 98, 100]:
        print("{}% of data is less than {} tokens.".format(p, np.percentile(token_length, p)))
    print()

    # character
    data["char_length"] = data["raw_text"].apply(lambda x: len(x))
    char_length = data["char_length"].to_numpy()
    for p in [50, 75, 80, 90, 95, 98, 100]:
        print("{}% of data is less than {} characters.".format(p, np.percentile(char_length, p)))

def main():
    analyze_length(phrase="train.all")
    analyze_length(phrase="valid")
    analyze_length(phrase="test")

if __name__ == "__main__":
    main()
