"""
This is the main part of the amazon product review crawler
"""
import os
import re
from bs4 import BeautifulSoup

class ProductInfo(object):
    def __init__(self, url = None):
        """

        :param url:
            URL example:
                url_d7200 =
                "https://www.amazon.com/Nikon-D7200-DX-format-DSLR-Black/dp/B00U2W45WA/ref=sr_1_3?s=electronics&ie=UTF8&qid=1525383460&sr=1-3&keywords=d7200&dpID=51nvvhu5VZL&preST=_SX300_QL70_&dpSrc=srch"
            URL format:
                "https://www.amazon.com/productName/.../id/..."
        asin, str: amazon product id
        product_name, str: amazon product name
        ratings, list of str(int): list of product ratings from 1-5 stars
        review_headlines, list of str: original headline of review
        review_texts, list of str: review text
        review_authors, list of str: author of reviews
        url, str: url of the product
        """
        # ex
        self.asin = None
        # ex
        self.product_name = None

        self.review_ratings = None
        self.review_headlines = None
        self.review_texts = None
        self.review_authors = None
        # ex
        self.url = url
        #self._get_asin(self.url)

    def _get_asin(self, url):
        """
        Get the asin(product id) of the amazon product
        :param url:
        :return:
        """
        # URL format: "https://www.amazon.com/productName/.../id/..."

        regex = re.compile(r'(?<=/)[^/]*')
        asin = regex.findall(url)[-2]
        if len(asin) != 10:
            # URL format https://www.amazon.com/.../id
            asin = regex.findall(url)[-1][:10]

        self.asin = asin
        #print("ASIN:{}".format(self.asin))

    def _delete_tmp_data_directory(self, tmp_data_directory):
        """
        delete tmp crawled data in the directory
        :param tmp_data_directory:
        :return:
        """
        path = tmp_data_directory + '/'
        if not os.path.exists(path):
            #print("---[Warning] No need to delete---")
            return
        pages = [file_ for file_ in os.listdir(path)]
        try:
            for page in pages:
                os.remove(path + page)
        except:
            pass
            #print("---[Warning] No files to delete---")
        try:
            os.rmdir(path)
        except:
            pass
            #print("---[Warning] No folder to delete!---")

    def crawl_raw_review_pages(self, review_crawl_number_limit = 100):
        """
        crawl amazon product's review pages.
        :param review_crawl_number_limit:
        :return:
        """
        try:
            self._get_asin(self.url)
        except:
            raise RuntimeError("---[Error] Wrong URL, no asin find---")

        tmp_data_root_directory = "../tmp_data/review/"
        tmp_data_directory =  '{}/com/{}'.format(tmp_data_root_directory,self.asin)

        #print("Directory: {}".format(tmp_data_directory))
        #print(os.path.abspath(tmp_data_directory))

        # if crawled before, delete it
        self._delete_tmp_data_directory(tmp_data_directory)
        # base cmd,

        cmd = 'python ./review_crawler/amazon_crawler.py'

        if not os.path.isdir(tmp_data_directory):
            try:
                """
                Sony Mp3
                python3 amazon_crawler.py -d com B0765ZVM6Y -m 100 -o reviews
                
                python3 amazon_crawler.py -d com B0765ZVM6Y -m 100 -o ../tmp_data/review

                Nikon D7200
                python3 amazon_crawler.py -d com B00U2W45WA -m 100 -o reviews       
                """

                newCmd = '{} -d com {} -m {} -o {}'.format(cmd, self.asin, review_crawl_number_limit, tmp_data_root_directory)
                #print("NEW CMD: {}".format(newCmd))
                #print("Running path:{}".format(__file__))
                os.system(newCmd)
            except NameError:
                raise RuntimeError('---[Error] Wrong Product ASIN---')

    def extract_product_review_info_from_crawled_page(self):

        tmp_data_root_directory = "../tmp_data/review/com"

        first_review_page_path = tmp_data_root_directory + '/{}/{}_1.html'.format(self.asin, self.asin)
        #print(first_review_page_path)
        # Solving the problem of encoding
        with open(first_review_page_path, 'r', encoding="utf-8") as html:
            soup = BeautifulSoup(html, 'html.parser')
            #soup = BeautifulSoup(html, 'lxml')

        try:
            self.product_name = soup.select('.a-link-normal')[0].text
        except:
            raise RuntimeError('---[Error] Wrong Review HTML Page, No Product Name field---')


        #print("Product Name: {}".format(self.product_name))


        tmp_data_directory = tmp_data_root_directory + '/{}/'.format(self.asin)
        review_pages = [file_ for file_ in os.listdir(tmp_data_directory) if file_[-5:] == '.html']

        review_ratings_list = []
        review_texts_list = []
        review_headlines_list = []
        review_authors_list = []
        for page in review_pages:
            with open(tmp_data_directory + page, 'r', encoding="utf-8") as f:
                soup = BeautifulSoup(f, 'html.parser')
                tags = soup.findAll("div", {"class": "a-section review"})
                if not tags:
                    print('{} is an invalid page format for scraping'.format(page))
                    continue

                for tag in tags:
                    review_text_class = "a-size-base review-text"
                    review_headline_class= "a-size-base a-link-normal review-title " \
                        "a-color-base a-text-bold"
                    review_author_class = "a-size-base a-link-normal author"
                    cur_review_rating = int(tag.find('i').text[0])
                    cur_review_text = tag.findAll("span", {"class": review_text_class})[0].text

                    try:
                        cur_review_author = tag.findAll("a", {"class": review_author_class})[0].text
                    except:
                        cur_review_author = 'anonymous'

                    try:
                        cur_review_headline = tag.findAll("a", {"class": review_headline_class})[0].text
                    except:
                        cur_review_headline = 'n/a'

                    review_authors_list.append(cur_review_author)
                    review_headlines_list.append(cur_review_headline)
                    review_texts_list.append(cur_review_text)
                    review_ratings_list.append(cur_review_rating)
        self.review_authors = review_authors_list
        self.review_headlines = review_headlines_list
        self.review_texts = review_texts_list
        self.review_ratings = review_ratings_list

