import sys
import time
import pathlib
from util.crawl_file_IO_utility import extract_asin
from review_crawler.crawler_driver import start_crawl
from review_summarize.review_analyze import analyze
from review_summarize.review_analyze_result_markdown_generate import generate_markdown_documnet

def start_process(product_url):
    print("[Info] Initializing...")
    t1 = time.time()

    product_asin = extract_asin(product_url)
    print("Product ASIN/ID: {}".format(product_asin))
    # Step 1: Create Output directory
    pathlib.Path('./output/{}'.format(product_asin)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('./output/{}/wordcloud_img'.format(product_asin)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('./output/{}/original_json_data'.format(product_asin)).mkdir(parents=True, exist_ok=True)

    t2 = time.time()
    #print("Time-Step1-{}".format(t2 - t1))
    print("\n[Info] Start Crawl")
    print("Crawling...")
    # Step 2: Crawl review page, generate json data
    start_crawl(product_url)

    t3 = time.time()
    print("[Info] Finish Crawl")
    print("Crawling Time Cost: {}s".format(t3- t1))

    print("\n[Info] Start Analyze and Summarize")
    print("Processing...")
    # Step 3:
    #       - Using trained model to summarize crawler review text
    #       - Sentiment analyze
    #       - Extract wordcloud
    #       - Extract neg/pos key features
    asin, product_name, mean_rating, positive_review_cnt, neutral_review_cnt, negative_review_cnt, positive_review_key_features, negative_review_key_features, pos_review_comprehensive_info_list, neg_review_comprehensive_info_list = analyze(product_url)

    t4 = time.time()
    print("[Info] Finish Crawl and Summarize")
    print("Analysis and Summarization Time Cost: {}s".format(t4 - t3))

    print("\n[Info] Start Generate Result Document")
    print('Generating...')
    # Step 4: Save result, generate markdown file
    generate_markdown_documnet(asin, product_name, mean_rating, positive_review_cnt, neutral_review_cnt, negative_review_cnt, positive_review_key_features, negative_review_key_features, pos_review_comprehensive_info_list, neg_review_comprehensive_info_list)

    t5 = time.time()
    print("[Info] Finish Generate")
    print("Result Document Generation Time Cost: {}s".format(t5 - t4))
    print("\nTotal Time Cost: {}s".format(t5 - t1))


def main():
    #url_sony_mp3 = "https://www.amazon.com/gp/product/B0765ZVM6Y/ref=s9_acsd_zwish_hd_bw_b1NbFQh_c_x_w?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-8&pf_rd_r=4GQHN4Y95PAR83QHEHC1&pf_rd_t=101&pf_rd_p=aaee489b-b256-5d30-b44e-827d0c6229b3&pf_rd_i=1264866011"

    #product_url = url_sony_mp3
    #GUI_Main(product_url)
    args = sys.argv
    if len(args) < 2:
        print("===[Error]-URL Missing")
    else:
        url = args[1]
        try:
            asin = extract_asin(url)
        except:
            print("===[Error]-Wrong URL===")
            return
        start_process(url)

        #print("URL:{}".format(url))


if __name__ == '__main__':
    main()
