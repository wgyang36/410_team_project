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


def loadReviewData(inputDirectoryPath, inputDataName):
    """
    Load UCSD review data
    :param inputDirectoryPath:
    :param inputDataName:
    :return:
    """
    JsonDataPath = inputDirectoryPath + "/" + inputDataName
    jsonData = load_JSON_file_lineByLine(JsonDataPath)
    return jsonData

def old_main():
    """
    Main Entrance
    :return:
    """
    input_Directory_Path_UCSD = "./input/410_DataSet/UCSD_DataSet/ucsd_standard_Json"

    #input_DataSet_Name_sd_reviews_Electronics_5 = "sd_reviews_Electronics_5.json"
    #input_UCSD_Data_sd_reviews_Electronics_5 = loadReviewData(input_Directory_Path_UCSD, input_DataSet_Name_sd_reviews_Electronics_5)

    input_DataSet_Name_sd_reviews_Cell_Phones_and_Accessories_5 = "sd_reviews_Cell_Phones_and_Accessories_5.json"
    input_UCSD_Data_sd_reviews_Cell_Phones_and_Accessories_5 = loadReviewData(input_Directory_Path_UCSD, input_DataSet_Name_sd_reviews_Cell_Phones_and_Accessories_5)

    #print("Total # of data: {}, Data Set Name: {}".format(len(input_UCSD_Data_sd_reviews_Electronics_5), input_DataSet_Name_sd_reviews_Electronics_5))
    #print("Total # of data: {}, Data Set Name: {}".format(len(input_UCSD_Data_sd_reviews_Cell_Phones_and_Accessories_5), input_DataSet_Name_sd_reviews_Cell_Phones_and_Accessories_5))
    """
    Total # of data: 1689188, Data Set Name: sd_reviews_Electronics_5.json
    Total # of data: 194439, Data Set Name: sd_reviews_Cell_Phones_and_Accessories_5.json
    """
