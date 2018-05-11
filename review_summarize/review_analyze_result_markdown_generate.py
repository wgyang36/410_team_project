import os
from util.crawl_file_IO_utility import extract_asin
from util.markdown_util import *


def construct_ulist_from_review_comprehensive_info(review_comprehensive_info, review_index):
    ulist = []
    ulist.append("Review Index: {}".format(review_index))
    ulist.append("Review Author: {}".format(review_comprehensive_info[1]))
    ulist.append("Review Rating: {} / 5".format(review_comprehensive_info[3]))
    ulist.append("Review Summary from Original Review: "+ review_comprehensive_info[2])
    ulist.append("Review Summary from Trained-model: "+ review_comprehensive_info[0])
    return ulist

def generate_markdown_documnet(product_asin, product_name, mean_rating, positive_review_cnt, neutral_review_cnt, negative_review_cnt, positive_review_key_features, negative_review_key_features, pos_review_comprehensive_info_list, neg_review_comprehensive_info_list):
    markdown_filepath = './output/{}/{}_review_analysis_result.md'.format(product_asin, product_asin)

    output_lines = []

    output_lines.append(header("{}-{} Review Analysis & Summarization Report".format(product_asin, product_name), 1))

    output_lines.append(header("Base Info", 2))
    output_lines.append(paragraph("Product Asin:{}".format(product_asin)))
    output_lines.append(paragraph("Product Name:{}".format(product_name)))
    output_lines.append(paragraph("Mean Rating:{} / 5".format(mean_rating)))
    output_lines.append(paragraph("Total Review Number: {}".format(int(positive_review_cnt) + int(neutral_review_cnt) + int(negative_review_cnt))))
    output_lines.append(paragraph("Positive Review Number:{}".format(positive_review_cnt)))
    output_lines.append(paragraph("Neutral Review Number:{}".format(neutral_review_cnt)))
    output_lines.append(paragraph("Negative Review Number:{}".format(negative_review_cnt)))

    output_lines.append(header("Review Sentiment Analysis & Summarization", 2))

    output_lines.append(header("Positive Reviews", 3))
    output_lines.append(header("Top Key Features of Positive Reviews", 4))
    output_lines.append(ulist(positive_review_key_features))
    output_lines.append(header("Wordcloud of Positive Reviews", 4))
    output_lines.append(image("Positive_Review_Wordcloud", "./wordcloud_img/{}_wordcloud_Positive_Reviews.png".format(product_asin)))
    output_lines.append(header("Summarization of Each Positive Reviews", 4))
    for i, pos_review_comprehensive_info in enumerate(pos_review_comprehensive_info_list):
        tmp_ulist = construct_ulist_from_review_comprehensive_info(pos_review_comprehensive_info, i)
        output_lines.append(ulist(tmp_ulist))
        output_lines.append('\n---')



    output_lines.append(header("Negative Reviews", 3))
    output_lines.append(header("Top Key Features of Negative Reviews", 4))
    output_lines.append(ulist(negative_review_key_features))
    output_lines.append(header("Wordcloud of Negative Reviews", 4))
    output_lines.append(image("Negative_Review_Wordcloud", "./wordcloud_img/{}_wordcloud_Negative_Reviews.png".format(product_asin)))
    output_lines.append(header("Summarization of Each Negative Reviews", 4))
    for i, neg_review_comprehensive_info in enumerate(neg_review_comprehensive_info_list):
        tmp_ulist = construct_ulist_from_review_comprehensive_info(neg_review_comprehensive_info, i)
        output_lines.append(ulist(tmp_ulist))
        output_lines.append('\n---')


    output_file = open(markdown_filepath, 'w', encoding = "utf-8")
    for line in output_lines:
        output_file.write(line + '\n')


    #output_file.writelines(output_lines)
    output_file.close()
    print("[Info] Result Have been Saved to: {}".format(os.path.abspath(markdown_filepath)))


def main():
    url_sony_mp3 = "https://www.amazon.com/gp/product/B0765ZVM6Y/ref=s9_acsd_zwish_hd_bw_b1NbFQh_c_x_w?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-8&pf_rd_r=4GQHN4Y95PAR83QHEHC1&pf_rd_t=101&pf_rd_p=aaee489b-b256-5d30-b44e-827d0c6229b3&pf_rd_i=1264866011"
    product_url = url_sony_mp3
    product_asin = extract_asin(product_url)
    generate_markdown_documnet(product_asin)
    pass

if __name__ == '__main__':
    main()