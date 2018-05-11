import json
import re
import string
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from util.crawl_file_IO_utility import read_json_file_with_URL
from review_summarize.review_feature_extraction import extract_key_features
from review_summarize.model_use import review_summarize_using_trained_model

def preprocess_review_text(original_review_text, stopwords_set):
    original_review_text = re.sub('[^a-z\s]', '', original_review_text.lower())
    original_review_text = [word for word in original_review_text.split() if word not in set(stopwords_set)]
    return ' '.join(original_review_text)

def get_stopwords_set():
    i = nltk.corpus.stopwords.words('english')
    # punctuations to remove
    j = list(string.punctuation)
    # finally let's combine all of these
    stopwords_set = set(i).union(j).union(('thiswas', 'wasbad', 'thisis', 'wasgood', 'isbad', 'isgood', 'theres', 'there'))
    return stopwords_set

def cal_sentiment_val(processed_review_text):
    sentiment = TextBlob(processed_review_text)
    return sentiment.sentiment.polarity


def save_wordcloud_plot(processed_review_text_list, figure_title, figure_save_path):
    wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=100, background_color='white', )
    wordcloud.generate(str(set(processed_review_text_list)))
    plt.figure(figsize=(20, 10))
    plt.title(figure_title, fontsize=40)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=10)
    #plt.show()
    plt.savefig(figure_save_path)


def get_review_info_list_and_dict(review_list, stopwords_set):
    review_text_list = []
    review_rating_list = []
    review_original_summary_list = []
    review_author_list = []
    review_text_index_dict = {}
    review_text_before_after_process_dict = {}
    for i, review in enumerate(review_list):
        review_text_list.append(review['review_text'])
        review_rating_list.append(review['rating'])
        review_original_summary_list.append(review['headline'])
        review_author_list.append(review['author'])
        key = review['review_text']
        value = i
        review_text_index_dict[key] = value
    processed_review_text_list = []
    for original_review_text in review_text_list:
        processed_review_text = preprocess_review_text(original_review_text, stopwords_set)
        processed_review_text_list.append(processed_review_text)
        review_text_before_after_process_dict[processed_review_text] = original_review_text

    return processed_review_text_list, review_text_list, review_rating_list, review_original_summary_list, review_author_list, review_text_index_dict, review_text_before_after_process_dict

def get_sub_sentiment_review_lists(processed_review_text_list):
    positive_processed_review_text_list = []
    neutral_processed_review_text_list = []
    negative_processed_review_text_list = []
    for processed_review_text in processed_review_text_list:
        sentiment_val = cal_sentiment_val(processed_review_text)
        if sentiment_val > 0:
            positive_processed_review_text_list.append(processed_review_text)
        elif sentiment_val == 0:
            neutral_processed_review_text_list.append(processed_review_text)
        else:
            negative_processed_review_text_list.append(processed_review_text)
    return positive_processed_review_text_list, neutral_processed_review_text_list, negative_processed_review_text_list

def get_mean_rating(review_rating_list):
    rating_sum = 0.0
    for review_rating in review_rating_list:
        review_rating_val = int(review_rating)
        rating_sum += review_rating_val
    return '%.3f' % (rating_sum / len(review_rating_list))

def get_tmp_file_path(product_asin):
    tmp_file_path = "../tmp_data/review/com/{}/".format(product_asin)
    return tmp_file_path

def get_original_review_elememt(text_after_summary_list, processed_review_text, processed_review_text_list, review_text_list, review_rating_list, review_original_summary_list, review_author_list, review_text_index_dict, review_text_before_after_process_dict):
    original_review_text = review_text_before_after_process_dict[processed_review_text]
    original_index = review_text_index_dict[original_review_text]

    original_review_rating = review_rating_list[original_index]
    original_review_author = review_author_list[original_index]
    original_review_summary = review_original_summary_list[original_index]
    summary_from_trained_model = text_after_summary_list[original_index]

    return summary_from_trained_model, original_review_author, original_review_summary, original_review_rating, original_review_text

def get_original_positive_negative_review_list(review_text_before_after_process_dict, positive_processed_review_text_list, neutral_processed_review_text_list, negative_processed_review_text_list):
    positive_original_review_text_list = []
    neutral_original_review_text_list = []
    negative_original_review_text_list = []

    for positive_processed_review_text in positive_processed_review_text_list:
        positive_original_review_text_list.append(review_text_before_after_process_dict[positive_processed_review_text])

    for neutral_processed_review_text in neutral_processed_review_text_list:
        neutral_original_review_text_list.append(review_text_before_after_process_dict[neutral_processed_review_text])

    for negative_processed_review_text in negative_processed_review_text_list:
        negative_original_review_text_list.append(review_text_before_after_process_dict[negative_processed_review_text])

    return positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list

def get_wordcloud_figure_path(asin, wordcloud_figure_title):
    wordcloud_figure_path = './output/{}/wordcloud_img/{}_wordcloud_{}'.format(asin, asin, wordcloud_figure_title)
    return wordcloud_figure_path

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
    json_file_path = "./output/{}/original_json_data/{}_reviews.json".format(product_asin, product_asin)
    #json_file_path = "../output/{}/original_json_data/{}_reviews.json".format(product_asin, product_asin)
    save_json_file(json_file_path, save_data)



def analyze(product_url):
    stopwords_set = get_stopwords_set()

    # Step 1: read json data
    product_review_info_data = read_json_file_with_URL(product_url)

    asin = product_review_info_data['asin']

    save_json_file_with_asin(product_review_info_data, asin)


    tmp_file_path = get_tmp_file_path(asin)
    product_name = product_review_info_data['product_name']
    review_list = product_review_info_data['review_list']
    processed_review_text_list, review_text_list, review_rating_list, review_original_summary_list, review_author_list, review_text_index_dict, review_text_before_after_process_dict = get_review_info_list_and_dict(review_list, stopwords_set)

    data_set_name = 'sd_reviews_Electronics_5'
    text_after_summary_length = 30
    batch_size = 64
    trained_model_name = 'mac_trained_model_batch_64'
    original_text_before_summary_list = review_text_list
    #text_after_summary_list = original_text_before_summary_list

    text_after_summary_list = review_summarize_using_trained_model(data_set_name, trained_model_name, batch_size, original_text_before_summary_list, text_after_summary_length)



    mean_rating = get_mean_rating(review_rating_list)
    #print("Mean Rating:{}".format(mean_rating))

    positive_processed_review_text_list, neutral_processed_review_text_list, negative_processed_review_text_list = get_sub_sentiment_review_lists(processed_review_text_list)

    positive_review_cnt = len(positive_processed_review_text_list)
    positive_wordcloud_title = "Positive_Reviews"
    positive_wordcloud_path = get_wordcloud_figure_path(asin, positive_wordcloud_title)
    #positive_wordcloud_path = tmp_file_path + '/{}_wordcloud_{}'.format(asin, positive_wordcloud_title)
    save_wordcloud_plot(positive_processed_review_text_list, positive_wordcloud_title, positive_wordcloud_path)

    neutral_review_cnt = len(neutral_processed_review_text_list)
    neutral_wordcloud_title = "Neutral_Reviews"
    neutral_wordcloud_path = get_wordcloud_figure_path(asin, neutral_wordcloud_title)
    #neutral_wordcloud_path = tmp_file_path + '/{}_wordcloud_{}'.format(asin, neutral_wordcloud_title)
    save_wordcloud_plot(neutral_processed_review_text_list, neutral_wordcloud_title, neutral_wordcloud_path)

    negative_review_cnt = len(negative_processed_review_text_list)
    negative_wordcloud_tile = "Negative_Reviews"
    negative_wordcloud_path = get_wordcloud_figure_path(asin, negative_wordcloud_tile)
    #negative_wordcloud_path = tmp_file_path + '/{}_wordcloud_{}'.format(asin, negative_wordcloud_tile)
    save_wordcloud_plot(negative_processed_review_text_list, negative_wordcloud_tile, negative_wordcloud_path)

    #print("Count. Pos:{}, Neu:{}, Neg:{}".format(positive_review_cnt, neutral_review_cnt, negative_review_cnt))

    positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list = get_original_positive_negative_review_list(review_text_before_after_process_dict, positive_processed_review_text_list, neutral_processed_review_text_list, negative_processed_review_text_list)
    positive_review_key_features, negative_review_key_features = extract_key_features(positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list)


    pos_review_comprehensive_info_list = []
    neg_review_comprehensive_info_list = []
    for pos_review in positive_processed_review_text_list:
        summary_from_trained_model, original_review_author, original_review_summary, original_review_rating, original_review_text = get_original_review_elememt(text_after_summary_list, pos_review, processed_review_text_list, review_text_list, review_rating_list, review_original_summary_list, review_author_list, review_text_index_dict, review_text_before_after_process_dict)
        pos_review_comprehensive_info_list.append([summary_from_trained_model,
                                               original_review_author,
                                               original_review_summary,
                                               original_review_rating,
                                               original_review_text])

    for neg_revie in negative_processed_review_text_list:
        summary_from_trained_model, original_review_author, original_review_summary, original_review_rating, original_review_text = get_original_review_elememt(text_after_summary_list, neg_revie, processed_review_text_list, review_text_list, review_rating_list, review_original_summary_list, review_author_list, review_text_index_dict, review_text_before_after_process_dict)
        neg_review_comprehensive_info_list.append([summary_from_trained_model,
                                               original_review_author,
                                               original_review_summary,
                                               original_review_rating,
                                               original_review_text])
    return asin, product_name, mean_rating, positive_review_cnt, neutral_review_cnt, negative_review_cnt, positive_review_key_features, negative_review_key_features, pos_review_comprehensive_info_list, neg_review_comprehensive_info_list

def main():
    url_sony_mp3 = "https://www.amazon.com/gp/product/B0765ZVM6Y/ref=s9_acsd_zwish_hd_bw_b1NbFQh_c_x_w?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-8&pf_rd_r=4GQHN4Y95PAR83QHEHC1&pf_rd_t=101&pf_rd_p=aaee489b-b256-5d30-b44e-827d0c6229b3&pf_rd_i=1264866011"
    product_url = url_sony_mp3
    analyze(product_url)

if __name__ == '__main__':
    main()