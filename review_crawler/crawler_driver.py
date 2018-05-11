from review_crawler.amazon_review_crawler import ProductInfo
from util.crawl_file_IO_utility import save_json_file_with_URL

def crawl_product_page(product_url, review_crawl_number_limit = 100):
    """
    get the raw review page
    :param product_url:
    :param review_crawl_number_limit:
    :return:
    """
    product_info = None
    product_info = ProductInfo(product_url)
    product_info.crawl_raw_review_pages(review_crawl_number_limit)
    return product_info

def extract_product_review_info(product_info):
    """
    extract product review info from the crawled pages
    :param product_info:
    :return:
    """
    product_info.extract_product_review_info_from_crawled_page()
    return product_info

def get_product_review_json_info(product_info):
    """
    transform product review info Class object to json object
    :param product_info:
    :return:
    """
    review_number = len(product_info.review_texts)
    review_data_list = []
    for i in range(review_number):

        review_data_list.append({
            'rating': product_info.review_ratings[i],
            'author': product_info.review_authors[i],
            'headline': product_info.review_headlines[i],
            'review_text': product_info.review_texts[i],
        })

    review_Data = {
        'asin': product_info.asin,
        'product_name': product_info.product_name,
        'review_list': review_data_list
    }
    return review_Data

def crawl_review_data(product_url):
    product_url = str(product_url.encode('utf-8'))

    # check if the url is empty
    if not product_url:
        raise RuntimeError("---[Error]-Empty Product URL---")
        #print("---[Error]-Empty Product URL---")
    print("***Main-Current Product URL:{}".format(product_url))

    """
    Crawler Parameter Setting
    crawl_review_number_limit: Number of reviews per product crawled
    """
    review_crawl_number_limit = 100
    product_info = crawl_product_page(product_url, review_crawl_number_limit)
    product_info = extract_product_review_info(product_info)

    product_review_json_info = get_product_review_json_info(product_info)
    save_json_file_with_URL(product_review_json_info, product_url)


def start_crawl(product_url):
    """
    driver of the crawler.
    :param product_url:
    :return:
    """
    product_url = str(product_url.encode('utf-8'))

    # check if the url is empty
    if not product_url:
        raise RuntimeError("---[Error]-Empty Product URL---")
        #print("---[Error]-Empty Product URL---")
    #print("***Main-Current Product URL:{}".format(product_url))

    """
    Crawler Parameter Setting
    crawl_review_number_limit: Number of reviews per product crawled
    """
    review_crawl_number_limit = 100
    product_info = crawl_product_page(product_url, review_crawl_number_limit)
    product_info = extract_product_review_info(product_info)

    product_review_json_info = get_product_review_json_info(product_info)
    save_json_file_with_URL(product_review_json_info, product_url)

def main():
    """
    Test URL 1
    Product Name:Nikon D7200
    Review Number: 366
    """
    url_d7200 = "https://www.amazon.com/Nikon-D7200-DX-format-DSLR-Black/dp/B00U2W45WA/ref=sr_1_3?s=electronics&ie=UTF8&qid=1525383460&sr=1-3&keywords=d7200&dpID=51nvvhu5VZL&preST=_SX300_QL70_&dpSrc=srch"
    """
    Test URL 2
    Product Name: Sony NW-A45/B Walkman with Hi-Res Audio, Grayish Black (2018 Model)
    Review Number: 17
    """
    url_sony_mp3 = "https://www.amazon.com/gp/product/B0765ZVM6Y/ref=s9_acsd_zwish_hd_bw_b1NbFQh_c_x_w?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-8&pf_rd_r=4GQHN4Y95PAR83QHEHC1&pf_rd_t=101&pf_rd_p=aaee489b-b256-5d30-b44e-827d0c6229b3&pf_rd_i=1264866011"
    start_crawl(url_sony_mp3)

if __name__ == '__main__':
    main()
