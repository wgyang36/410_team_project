"""
transform the UCSD amazon review data to standard json format
"""
import json
import csv
import gzip
from util.summarization_part_util import load_UCSD_reviewData


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

def transform_to_standard_JSON():
    inputPath = './input/410_DataSet/UCSD_DataSet' + "/original/" + "meta_Cell_Phones_and_Accessories.json.gz"
    outputPath = './input/410_DataSet/UCSD_DataSet' + "/ucsd_standard_Json/" + "sd_meta_Cell_Phones_and_Accessories.json"

    f = open(outputPath, 'w')
    for l in parse(inputPath):
        f.write(l + '\n')

def write_to_csv(json_data, fileName):
    file_path = '../input/ucsd_data_headline_text_csv' + '/{}.csv'.format(fileName)
    print(file_path)
    f = csv.writer(open(file_path, "w"))
    f.writerow(["review_summary", "review_text"])
    for cur_row_data in json_data:
        f.writerow([cur_row_data['review_summary'],
                    cur_row_data['review_text']])

def transform_json_to_csv():
    test_name = 'sd_reviews_Cell_Phones_and_Accessories_5'
    train_name = 'sd_reviews_Electronics_5'

    fileName = test_name
    json_data = load_UCSD_reviewData(fileName + '.json')
    print(len(json_data))
    new_json_data = []
    for curData in json_data:
        new_json_data.append({
            'review_summary': curData['summary'],
            'review_text': curData['reviewText']
        })
    write_to_csv(new_json_data, fileName)



def main():
    transform_json_to_csv()
if __name__ == '__main__':
    main()