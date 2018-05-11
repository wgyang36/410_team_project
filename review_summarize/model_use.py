import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from review_summarize.clean_data import clean_text
from util.summarization_part_util import load_pickled_data
def text_to_seq(text, vocabulary_to_int):
    """
    transform the text to int values so that we could use the model to handle it.
    :param text:
    :param vocabulary_to_int:
    :return:
    """
    text = clean_text(text)
    return [vocabulary_to_int.get(word, vocabulary_to_int['<UNK>']) for word in text.split()]


def summary_original_text_list(trained_model_name, text_after_summary_length, vocabulary_to_int, int_to_vocabulary, batch_size, original_text_before_summary_list):
    """
    :param trained_model_name:
    :param text_after_summary_length:
    :param vocabulary_to_int:
    :param int_to_vocabulary:
    :param batch_size:
    :param original_text_before_summary_list:
    :return:
    """
    text_before_summary_list = [text_to_seq(original_text_before_summary, vocabulary_to_int) for
                                original_text_before_summary in original_text_before_summary_list]
    text_after_summary_list = []
    checkpoint = "./trained_model/{}/best_model.ckpt".format(trained_model_name)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        # Multiply by batch_size to match the model's input parameters
        for i, text in enumerate(text_before_summary_list):
            answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                              summary_length: [text_after_summary_length],
                                              # summary_length: [np.random.randint(5,8)],
                                              text_length: [len(text)] * batch_size,
                                              keep_prob: 1.0})[0]
            pad = vocabulary_to_int["<PAD>"]
            text_after_summary = " ".join([int_to_vocabulary[i] for i in answer_logits if i != pad])
            text_after_summary_list.append(text_after_summary)
    return text_after_summary_list

def review_summarize_using_trained_model(data_set_name, trained_model_name, batch_size, original_text_before_summary_list, text_after_summary_length):
    saved_data_root_directory = './tmp_data/summarization_data/' + data_set_name
    vocabulary_to_int = load_pickled_data(saved_data_root_directory + '/vocabulary_to_int.p')
    int_to_vocabulary = load_pickled_data(saved_data_root_directory + '/int_to_vocabulary.p')

    text_after_summary_list = summary_original_text_list(trained_model_name, text_after_summary_length, vocabulary_to_int, int_to_vocabulary, batch_size, original_text_before_summary_list)

    return text_after_summary_list




def main():
    uscd_Electronic = 'sd_reviews_Electronics_5'
    data_set_name = uscd_Electronic

    model_name_1 = 'mac_trained_model_batch_64'
    model_name_2 = 'model_0.951_batch_64_60000'
    trained_model_name = model_name_1

    s1 = "This is my favorite DX camera to date and contrary to popular beliefs, the D7200 is a massive upgrade over the D7100. Low light and high ISO settings performance are much better and the focusing system is extremely good (more accurate focusing over the D7100). I take this camera everywhere I go and just love the photos coming out of this camera. This camera has the performance of Nikon's full frame cameras, but at the price of a cropped sensor camera. All the controls and dials are very well thought out and it won't be a stranger if you're used to Nikon D610 and D750. Unless you need the ultimate low light performance of a full frame camera (which is marginally better than this camera), then the D7200 is the right camera for a semi-pro shooter. I ended up leaving my Nikon D810 at home most of the time and this little guy gets to travel with me 90% of the time. Just get some good lenses for it and you will be amazed by the quality of this camera."
    s2 = "This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!"
    original_text_before_summary_list = ["The coffee tasted great and was at such a good price! I highly recommend this to everyone!",
                        "love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets",
                       s1,
                       s2
                       ]
    text_after_summary_length = 30
    batch_size = 64
    text_after_summary_list = review_summarize_using_trained_model(data_set_name, trained_model_name, batch_size,
                                             original_text_before_summary_list, text_after_summary_length)

    for i, original_text_before_summary in enumerate(original_text_before_summary_list):
        text_after_summary = text_after_summary_list[i]
        print("*"*10)
        print("Original Text:")
        print(original_text_before_summary)
        print("After Summary:")
        print(text_after_summary)
        print("="*10)


if __name__ == '__main__':
    main()