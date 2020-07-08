import ujson as json
import os, os.path

def predict_class():
    filename = "/apple_data/workspace/location/model/proposed_sum_city/predict_e60.json"
    filename = "/apple_data/workspace/location/model/cnn3/predict_e5.json"
    filename = "/apple_data/workspace/location/model/proposed_reg_position_sum_city/predict_e3.json"
    #filename = "/apple_data/workspace/location/model/proposed_reg_position_sum_city_selu/predict_e12.json"
    
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
   
    data = set(data)
    print("total num = ", len(data))
    
    data = sorted(list(data))
    print(data) 

def compute_country_acc():
    dir_path = "/apple_data/workspace/location/model/lab_cnn_pure_textcity"
    epoch = 17
    class_dictionary = os.path.join(dir_path, "class_dictionary")
    filename = os.path.join(dir_path, "predict_e{}.json".format(epoch))

    with open(class_dictionary, "r", encoding='utf-8') as infile:
        class_dict = json.load(infile)
        class_dict = {
            v:k
            for k, v in class_dict.items()        
        }

    with open(filename, 'r', encoding='utf-8') as infile:
        results = json.load(infile)

    city_correct = []
    country_correct = []
    for p, y in zip(results["predict"], results["y_labels"]):
        p_city = class_dict[p]
        y_city = class_dict[y]

        p_country = p_city.split("-")[-1]
        y_country = y_city.split("-")[-1]

        if p_country == y_country:
            country_correct.append(1)
        else:
            country_correct.append(0)

        if p_city == y_city:
            city_correct.append(1)
        else:
            city_correct.append(0)

    print("city_acc = {}".format(sum(city_correct)/len(city_correct)))
    print("country_acc = {}".format(sum(country_correct)/len(country_correct)))

if __name__ == "__main__":
    #predict_class()
    compute_country_acc()
