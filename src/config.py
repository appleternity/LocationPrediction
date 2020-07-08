import os, os.path

verbose = True
process_data = False

# path
## for quicikly test the model
#train_data = "/apple_data/workspace/IE/data/WNUT/data/valid_cnn.tweet.json"
#train_label = "/home/appleternity/workspace/lab/graph/baseline/twitter-deepgeo/data/valid/label.tweet.json"
train_data  = "/apple_data/workspace/IE/data/WNUT/data/train_cnn.tweet.json"
train_label = "/apple_data/workspace/IE/data/WNUT/data/train.label.json"

valid_data = "/apple_data/workspace/IE/data/WNUT/data/valid_cnn.tweet.json"
valid_label = "/home/appleternity/workspace/lab/graph/baseline/twitter-deepgeo/data/valid/label.tweet.json"

test_data = "/apple_data/workspace/IE/data/WNUT/data/test_cnn.tweet.json"
test_label   = "/home/appleternity/workspace/lab/graph/baseline/twitter-deepgeo/data/test/label.tweet.json"
#test_data = "/apple_data/workspace/IE/data/WNUT/data/valid_cnn.tweet.json"
#test_label = "/home/appleternity/workspace/lab/graph/baseline/twitter-deepgeo/data/valid/label.tweet.json"

# dictionary
city = "city"
model_dir = "/apple_data/workspace/location/model/new_proposed_nonreg_join_position_sum_{}_joint_weight4".format(city)
word_dictionary_path = os.path.join(model_dir, "word_dictionary.json")
class_dictionary_path = os.path.join(model_dir, "class_dictionary.json")
meta_dictionary_path = os.path.join(model_dir, "meta_dictionary.json")
history_path = os.path.join(model_dir, "history.json")

train_file = os.path.join(model_dir, "train.h5")
valid_file = os.path.join(model_dir, "valid.h5")
test_file = os.path.join(model_dir, "test.h5")

try:
    os.mkdir(model_dir)
except:
    pass

# training
keep_training = False
start_epoch = 0

# testing
testing_epoch = 13

# parameter setting
vocab_size = None
minfreq = 10
emb_dim = 200
char_dim = 100
hidden_dim = 200
char_hidden_dim = 100
#num_head = 10
num_head = 2
#char_num_head = 8
char_num_head = 2
text_len = 30
char_len = 140
output_size = 10
#filter_list = {3:128, 4:128, 5:64, 6:64, 7:64}
filter_list = {3:64, 4:64, 5:64, 6:64, 7:64}
#filter_list = {3:128, 4:128, 5:128, 6:128, 7:128}
#filter_list = {3:64, 4:64, 5:64}
#filter_list = {3:64, 5:64, 7:64}

#dropout_rate = 0.7
dropout_rate = 0.3
#learning_rate = 0.0005
learning_rate = 0.001
batch_size = 512
#batch_size = 1024
epochs = 10
reg = False
reg_weight = 0.01
geo_cord = False
meta_feature = False
meta_dim = 50
