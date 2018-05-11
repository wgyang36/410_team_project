"""
Utilities for the review_summary package part
Serialization and de-serialization data
"""
import pickle
import json
def pickle_data(file_path, data):
    """
    save data to disk
    :param file_path:
    :param data:
    :return:
    """
    with open(file_path, 'wb') as save_file:
        pickle.dump(data, save_file)
        #pickle.dump(save_file, data)

def load_pickled_data(file_path):
    """
    load saved serialization data from disk
    :param file_path:
    :return:
    """
    data = None
    with open(file_path, 'rb') as saved_file:
        data = pickle.load(saved_file)
    return data

def load_JSON_file_OnceWhole(JsonFilePath):
    """
    Load JSON File as a whole
    :param JsonFilePath:
    :return:
    """
    data = None
    with open(JsonFilePath) as data_file:
        for line in data_file:
            data = json.load(line)
    return data

def load_JSON_file_lineByLine(JsonFilePath):
    """
    Load JSON file line by line
    One JSON object per line
    :param JsonFilePath:
    :return:
    """
    data = []
    with open(JsonFilePath) as data_file:
        for line in data_file:
            data.append(json.loads(line))
    return data


def load_UCSD_reviewData(ucsd_data_name):
    """
    Load UCSD review data
    :param inputDirectoryPath:
    :param inputDataName:
    :return:
    """
    test_name = 'sd_reviews_Cell_Phones_and_Accessories_5.json'
    train_name = 'sd_reviews_Electronics_5.json'
    ucsd_data_root_directory = "../input/410_DataSet/UCSD_DataSet/ucsd_standard_Json"
    json_data_path = ucsd_data_root_directory + "/" + ucsd_data_name

    jsonData = load_JSON_file_lineByLine(json_data_path)
    return jsonData
