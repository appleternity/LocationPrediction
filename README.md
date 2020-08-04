# Location Prediction for Tweets
This repo contains code for the paper **Location Prediction for Tweets** [https://www.frontiersin.org/articles/10.3389/fdata.2019.00005/full]. 

## Data
Please follow the instruction from WNUT 2016 Geo-Shared-Task to acquire the data.

WNUT 2016 Geo-Shared-Task: https://noisy-text.github.io/2016/geo-shared-task.html

## Path Configuration
Before running the code, please modify the path settint in the **src/config.py** file.
```python
root_dir = "PATH_TO_THE_GIT_REPO"
```

## Development Environment
Python3.6 + Tensorflow 1.12.0

## Data Preprocessing
Take the downloaded tweets and the label file, extract the needed data field, perform tokenization on the tweets, and aggregrate the ground truth.

```console
$ python preprocessing.py --tweet_path ../data/train_tweet.json --label_path ../data/train.label.json --output_path ../data/train.parquet
$ python preprocessing.py --tweet_path ../data/valid_tweet.json --label_path ../data/valid.label.json --output_path ../data/valid.parquet
$ python preprocessing.py --tweet_path ../data/test_tweet.json --label_path ../data/test.label.json --output_path ../data/test.parquet
```

## Model Training
The script for training the model from scratch. The default hyperparameter is used in the paper so you can use the following command to start the training.
```console
$ python train.py
```

If you would like to change the hyperparameter, please refer to the following argument settings.
```console
$ python train.py [-h] [--max_len MAX_LEN] [--max_char_len MAX_CHAR_LEN]
                [--minfreq MINFREQ] [--emb_dim EMB_DIM]
                [--hidden_dim HIDDEN_DIM] [--num_head NUM_HEAD]
                [--layer_num LAYER_NUM] [--char_dim CHAR_DIM]
                [--char_hidden_dim CHAR_HIDDEN_DIM]
                [--char_num_head CHAR_NUM_HEAD]
                [--char_layer_num CHAR_LAYER_NUM] [--filter FILTER_LIST]
                [--dropout_rate DROPOUT_RATE] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--reg REG]
                [--reg_weight REG_WEIGHT] [--data_redo DATA_REDO]
                [--note NOTE] [--gpu GPU] [--train_data TRAIN_DATA]
```

#### optional arguments:
| Argument            | Value                                      | Information                                              |
|---------------------|--------------------------------------------|----------------------------------------------------------|
| -h, --help          |                                            | show this help message and exit                          |
| --max_len           | INT (Default:30)                           | maximum length of the tokens                             |
| --max_char_len      | INT (Default: 140)                         | maximum length of the characters                         |
| --minfreq           | INT (Default: 10)                          | minimum frequency of the vocabulary and character        |
| --emb_dim           | INT (Default: 200)                         | word embedding dimension                                 |
| --hidden_dim        | INT (Default: 200)                         | hidden dimension                                         |
| --num_head          | INT (Default: 10)                          | number of head of the transformer                        |
| --layer_num         | INT (Default: 2)                           | number of layer of the transformer                       |
| --char_dim          | INT (Default: 100)                         | character embedding dimension                            |
| --char_hidden_dim   | INT (Default: 100)                         | character hidden dimension                               |
| --char_num_head     | INT (Default: 8)                           | number of head of the character transformer              |
| --char_layer_num    | INT (Default: 2)                           | number of layer of the character transformer             |
| --filter            | STRING <br>(Default: 3:64-4:64-5:64-6:64-7:64) | filter configuration of the character CNN, ex: 3:64-4:64 |
| --dropout_rate      | FLOAT (Default: 0.3)                       | dropout rate across the model                            |
| --learning_rate     | FLOAT (Default: 1e-4)                      | learning rate                                            |
| --batch_size        | INT (Default: 128)                         | batch size                                               |
| --epochs            | INT (Default: 30)                          | number of epochs for training                            |
| --reg               | BOOL (Default: False)                      | whether use regularizer or not                           |
| --reg_weight        | FLOAT (Default: 1e-4)                      | weighting for regularizer                                |
| --data_redo         | BOOL (Default: False)                      | re-process the data again                                |
| --note              | STRING (Default: "")                       | note for the model name                                  |
| --gpu               | STRING (Default: "0")                      | gpu setting                                              |
| --train_data        | STRING (Default: "train")                  | filename of the training data                            |


## Model Testing
```console
$ python test.py [-h] --model_folder MODEL_FOLDER --target_epoch TARGET_EPOCH                                             
               [--gpu GPU]
```

#### optional arguments:
| Argument            | Value                                      | Information                                              |
|---------------------|--------------------------------------------|----------------------------------------------------------|
| -h, --help          |                                            | show this help message and exit                          |
| --model_folder      | STRING                                     | the path to the target model's folder                    |
| --target_epoch      | STRING                                     | specify the model for testing                            |
| --gpu               | STRING (Default: "0")                      | gpu setting                                              |


## Inferencing
To inference location (city & country) using the trained model, you will need to first process the input file into the required format where each line represents a sample.
The output will be stored as a CSV file with three columns, text, city, and country.

```console
$ python inference.py [-h] --model_folder MODEL_FOLDER --target_epoch
                    TARGET_EPOCH --text_file TEXT_FILE --output_file
                    OUTPUT_FILE [--gpu GPU]
```

#### optional arguments:
| Argument       | Value                 | Information                           |
|----------------|-----------------------|---------------------------------------|
| -h, --help     |                       | show this help message and exit       |
| --model_folder | STRING                | the path to the target model's folder |
| --target_epoch | STRING                | specify the model for testing         |
| --text_file    | STRING                | the path to the testing text file     |
| --output_file  | STRING                | the path to the output file           |
| --gpu          | STRING (Default: "0") | gpu setting                           |


## Trained Model
Please find the release trained model here.

https://drive.google.com/file/d/1M8AxKuVmwRM3jEVk3iYH0BEmKOKsr_zP/view?usp=sharing

The performance of the released model is as follow. 
|               | City Acc | Country Acc |
|---------------|----------|-------------|
| Release Model | 0.2163   | 0.6110      |

Uncompress the .tar file after downloading the model.
```console
$ tar xvf release.tar
```

You can run the inference script by using the following command.
```console
$ python inference.py --model_folder ../model/release --target_epoch 1 --text_file ../sample_text/sample.txt --output_file ../sample_text/output.csv --gpu 6
```

Here, I have my folders in the following structure.
```
root_dir (LocationPrediction)
| - src
| - model
| | - release
| - sample_text
```

## Any Questions?
Please sent me an email at chiehyang@psu.edu

## Citation
Please cite the following papers if you use this repo for testing, auto geo-labeling, or comparison.
```bibtex
@ARTICLE{10.3389/fdata.2019.00005,
  AUTHOR={Huang, Chieh-Yang and Tong, Hanghang and He, Jingrui and Maciejewski, Ross},   
  TITLE={Location Prediction for Tweets},      
  JOURNAL={Frontiers in Big Data},      
  VOLUME={2},     
  PAGES={5},     
  YEAR={2019},      
  URL={https://www.frontiersin.org/article/10.3389/fdata.2019.00005},       
  DOI={10.3389/fdata.2019.00005},      
  ISSN={2624-909X},   
  ABSTRACT={Geographic information provides an important insight into many data mining and social media systems. However, users are reluctant to provide such information due to various concerns, such as inconvenience, privacy, etc. In this paper, we aim to develop a deep learning based solution to predict geographic information for tweets. The current approaches bear two major limitations, including (a) hard to model the long term information and (b) hard to explain to the end users what the model learns. To address these issues, our proposed model embraces three key ideas. First, we introduce a multi-head self-attention model for text representation. Second, to further improve the result on informal language, we treat subword as a feature in our model. Lastly, the model is trained jointly with the city and country to incorporate the information coming from different labels. The experiment performed on W-NUT 2016 Geo-tagging shared task shows our proposed model is competitive with the state-of-the-art systems when using accuracy measurement, and in the meanwhile, leading to a better distance measure over the existing approaches.}
}
```
