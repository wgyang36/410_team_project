import re
import pandas as pd
import numpy as np
import time
from nltk.corpus import stopwords
from util.summarization_part_util import pickle_data

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there'd": "there had",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where'd": "where did",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are"
        }

def clean_text(text, remove_stopwords=True):
    """
    Remove
        - unwanted characters
        - stopwords,
        - format the text to create fewer nulls word embeddings
    :param text:
    :param remove_stopwords:
    :return:
    """

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        # We are not using "text.split()" here
        # since it is not fool proof, e.g. words followed by punctuations "Are you kidding?I think you aren't."
        text = re.findall(r"[\w']+", text)
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # remove links
    text = re.sub(r'\<a href', ' ', text)  # remove html link tag
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text

def get_cleaned_review_summary_and_texts(review_data):
    """
    clean stopword, unrelated info in the review summary and review text.
    :param review_data:
    :return:
    """
    cleaned_review_summaries = []
    cleaned_review_texts = []
    for summary in review_data.review_summary:
        cleaned_review_summaries.append(clean_text(summary, remove_stopwords=False))

    for text in review_data.review_text:
        cleaned_review_texts.append(clean_text(text, remove_stopwords=True))

    return cleaned_review_summaries, cleaned_review_texts

def count_word(word_count_dict, text):
    """
    count each word's occurrence in the review data
    :param word_count_dict:
    :param text:
    :return:
    """
    for sentence in text:
        for word in sentence.split():
            if word not in word_count_dict:
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1

def load_conceptnet_numberbatch_embeddings(embeddings_index):
    """
    https://github.com/commonsense/conceptnet-numberbatch
    :param embeddings_index:
    :return:
    """
    embedding_file_path = '../assert/numberbatch-en-17.06.txt'
    with open(embedding_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

def count_missing_word_num(word_count_dict, embeddings_index):
    """
    Count # of word missing in Conceptnet Numberbatch
    :param word_count_dict:
    :param embeddings_index:
    :return:
    """
    missing_word_num = 0
    threshold = 20
    for word, count in word_count_dict.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_word_num += 1
    return missing_word_num

def convert_vocabulary_to_int(vocabulary_to_int, word_count_dict, embeddings_index, threshold):
    # Index words from 0
    index = 0
    for word, count in word_count_dict.items():
        if count >= threshold or word in embeddings_index:
            vocabulary_to_int[word] = index
            index += 1
    # Special token
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]

    for code in codes:
        vocabulary_to_int[code] = len(vocabulary_to_int)

def convert_int_to_vocabulary(vocabulary_to_int, int_to_vocabulary):
    for word, value in vocabulary_to_int.items():
        int_to_vocabulary[value] = word

def create_wording_embedding_matrix(vocabulary_to_int, embeddings_index):
    embedding_dim = 300
    nb_words = len(vocabulary_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocabulary_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    return word_embedding_matrix

def convert_sentence_to_ints (text, word_count, unk_count, vocabulary_to_int, eos=False):
    """
    Convert words in text to an integer.If word is not in vocab_to_int, use UNK's integer.
    :param text:
    :param word_count:
    :param unk_count:
    :param eos:
    :return:
    """
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocabulary_to_int:
                sentence_ints.append(vocabulary_to_int[word])
            else:
                sentence_ints.append(vocabulary_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocabulary_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

def unk_counter(sentence, vocabulary_to_int):
    """
    Counts the number of time UNK appears in a sentence.
    :param sentence:
    :return:
    """
    unk_count = 0
    for word in sentence:
        if word == vocabulary_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

def filter_condition(item, vocabulary_to_int):
    max_text_length = 83
    max_summary_length = 13
    min_length = 2
    # text can contain up to 1 UNK word
    unk_text_limit = 1
    # Summary should not contain any UNK word
    unk_summary_limit = 0

    int_summary = item[0]
    int_text = item[1]
    if(len(int_summary) >= min_length and
       len(int_summary) <= max_summary_length and
       len(int_text) >= min_length and
       len(int_text) <= max_text_length and
       unk_counter(int_summary, vocabulary_to_int) <= unk_summary_limit and
       unk_counter(int_text, vocabulary_to_int) <= unk_text_limit):
        return True
    else:
        return False

def main(csv_file_path, data_name):
    t1 = time.time()
    # Step 1: load csv Data, clean null value, reset index
    review_data = pd.read_csv(csv_file_path)
    review_data = review_data.dropna()
    review_data = review_data.reset_index(drop=True)

    curTime = time.time()
    print("Time: Step1: {}s".format(curTime - t1))
    t2 = time.time()

    # Step 2: remove stop words, and other unrelated words
    cleaned_review_summaries, cleaned_review_texts = get_cleaned_review_summary_and_texts(review_data)


    curTime = time.time()
    print("Time: Step2: {}s".format(curTime - t2))
    t3 = time.time()

    # Step 3: review word usage statistics
    word_count_dict = {}
    count_word(word_count_dict, cleaned_review_summaries)
    count_word(word_count_dict, cleaned_review_texts)


    curTime = time.time()
    print("Time: Step3: {}s".format(curTime - t3))
    t4 = time.time()
    # Step 4: Load conceptnet numberbatch's embeddings
    embeddings_index = {}
    load_conceptnet_numberbatch_embeddings(embeddings_index)
    threshold = 20
    missing_word_num = count_missing_word_num(word_count_dict, embeddings_index)

    curTime = time.time()
    print("Time: Step4: {}s".format(curTime - t4))
    t5 = time.time()
    # Step 5: create word to int, int to word dicts
    # dict: convert word to int
    vocabulary_to_int = {}  # Index words from 0
    convert_vocabulary_to_int(vocabulary_to_int, word_count_dict, embeddings_index, threshold)
    # dict: convert int to word
    int_to_vocabulary = {}
    convert_int_to_vocabulary(vocabulary_to_int, int_to_vocabulary)


    curTime = time.time()
    print("Time: Step5: {}s".format(curTime - t5))
    t6 = time.time()
    # Step 6: create word embedding matrix
    word_embedding_matrix = create_wording_embedding_matrix(vocabulary_to_int, embeddings_index)


    curTime = time.time()
    print("Time: Step6: {}s".format(curTime - t6))
    t7 = time.time()
    # Step 7: Apply convert_sentence_to_int on summary and text
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_sentence_to_ints(cleaned_review_summaries, word_count, unk_count, vocabulary_to_int)
    int_texts, word_count, unk_count = convert_sentence_to_ints(cleaned_review_texts, word_count, unk_count, vocabulary_to_int, eos=True)

    curTime = time.time()
    print("Time: Step7: {}s".format(curTime - t7))
    t8 = time.time()
    # Step 8:
    int_text_summaries = list(zip(int_summaries, int_texts))
    int_text_summaries_filtered = list(filter(lambda x: filter_condition(x, vocabulary_to_int), int_text_summaries))
    #int_text_summaries_filtered = list(filter(filter_condition, int_text_summaries))
    sorted_int_text_summaries = sorted(int_text_summaries_filtered, key=lambda item: len(item[1]))
    sorted_int_text_summaries = list(zip(*sorted_int_text_summaries))

    sorted_review_summaries = list(sorted_int_text_summaries[0])
    sorted_review_texts = list(sorted_int_text_summaries[1])

    curTime = time.time()
    print("Time: Step8: {}s".format(curTime - t8))
    t9 = time.time()
    # Step 9: save cleaned data
    saved_data_root_directory = "../tmp_data/summarization_data"


    pickle_data(saved_data_root_directory + '/{}/cleaned_review_summaries.p'.format(data_name),cleaned_review_summaries)
    pickle_data(saved_data_root_directory + '/{}/cleaned_review_texts.p'.format(data_name),cleaned_review_texts)

    pickle_data(saved_data_root_directory + '/{}/sorted_summaries.p'.format(data_name),sorted_review_summaries)
    pickle_data(saved_data_root_directory + '/{}/sorted_review_texts.p'.format(data_name),sorted_review_texts)
    pickle_data(saved_data_root_directory + '/{}/word_embedding_matrix.p'.format(data_name),word_embedding_matrix)

    pickle_data(saved_data_root_directory + '/{}/vocabulary_to_int.p'.format(data_name),vocabulary_to_int)
    pickle_data(saved_data_root_directory + '/{}/int_to_vocabulary.p'.format(data_name),int_to_vocabulary)

    curTime = time.time()
    print("Time: Step9: {}s".format(curTime - t9))

if __name__ == '__main__':

    uscd_Electronic_csv = 'sd_reviews_Electronics_5'
    ucsd_Cell_Phones_and_Accessories_5_csv = 'sd_reviews_Cell_Phones_and_Accessories_5'

    data_name = uscd_Electronic_csv
    csv_file_path = '../input/ucsd_data_headline_text_csv/{}.csv'.format(data_name)

    main(csv_file_path, data_name)
    """
    Time: Step1: 14.822831869125366s
    Time: Step2: 449.56904697418213s
    Time: Step3: 40.49072623252869s
    Time: Step4: 26.848496198654175s
    Time: Step5: 0.1682729721069336s
    Time: Step6: 0.41628503799438477s
    Time: Step7: 51.46222901344299s
    Time: Step8: 13.008572816848755s
    Time: Step9: 5.179518938064575s
    
    Process finished with exit code 0

    """