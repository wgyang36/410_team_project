import json
import re

def extract_asin(product_url):
    """
    Get the asin(product id) of the amazon product
    :param url:
    :return:
    """
    # URL format: "https://www.amazon.com/productName/.../id/..."
    url = product_url
    regex = re.compile(r'(?<=/)[^/]*')
    asin = regex.findall(url)[-2]
    if len(asin) != 10:
        # URL format https://www.amazon.com/.../id
        asin = regex.findall(url)[-1][:10]
    #print("ASIN:{}".format(asin))
    return asin

def get_json_file_path(product_asin):
    """
    combination of json file path
    :param product_asin:
    :return:
    """
    json_file_path = "../tmp_data/review/com/{}/{}_reviews.json".format(product_asin, product_asin)
    return json_file_path

def save_json_file(save_path, save_data):
    """
    save data to the path in the json format
    :param save_path:
    :param save_data:
    :return:
    """
    with open(save_path, 'w') as fp:
        json.dump(save_data, fp, sort_keys = True, indent = 4)

def save_json_file_with_asin(save_data, product_asin):
    json_file_path = "../output/{}/original_json_data/{}_reviews.json".format(product_asin, product_asin)
    save_json_file(json_file_path, save_data)


def save_json_file_with_URL(save_data, product_url):
    """
    save data to path according to the product url
    :param save_data:
    :param product_url:
    :return:
    """
    product_asin = extract_asin(product_url)
    save_path = get_json_file_path(product_asin)
    save_json_file(save_path, save_data)

def read_json_file(file_path):
    """
    read product review json file
    :param file_path:
    :return:
    """
    json_data = None
    with open (file_path) as data_file:
        json_data = json.load(data_file)
    return json_data

def read_json_file_with_asin(product_asin):
    """
    read json file according to the product asin
    :param product_asin:
    :return:
    """
    read_path = get_json_file_path(product_asin)
    return read_json_file(read_path)

def read_json_file_with_URL(product_url):
    """
    read json file according to the product url
    :param product_url:
    :return:
    """
    product_asin = extract_asin(product_url)
    product_review_info_data = read_json_file_with_asin(product_asin)
    return product_review_info_data
