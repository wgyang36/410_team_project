"""
This code file credits to https://github.com/rahulreddykr/Review_Summarization-Aspect_based_opinion_mining/blob/master/ReviewAnalysisFinal.ipynb
We modified it to suit our program.
"""
import nltk
import inflect

def tokenize_review_sentences(pos_sentences, neutral_sentences, neg_sentences):
    pos_tokens = []
    pos_tokens_postagged = []
    neg_tokens = []
    neg_tokens_postagged = []
    neut_tokens = []
    neut_tokens_postagged = []
    '''Tokenize the sentences into words'''
    for sent in pos_sentences:
        pos_tokens.append(nltk.word_tokenize(sent))
    for sent in neg_sentences:
        neg_tokens.append(nltk.word_tokenize(sent))
    for sent in neutral_sentences:
        neut_tokens.append(nltk.word_tokenize(sent))

    '''Apply Part of speech tagging for the tokenized words'''
    for sent in pos_tokens:
        pos_tokens_postagged.append(nltk.tag.pos_tag(sent))
    for sent in neg_tokens:
        neg_tokens_postagged.append(nltk.tag.pos_tag(sent))
    for sent in neut_tokens:
        neut_tokens_postagged.append(nltk.tag.pos_tag(sent))
    return pos_tokens, pos_tokens_postagged, neut_tokens, neut_tokens_postagged, neg_tokens, neg_tokens_postagged


def acceptable_word(word, stopwords):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool((2 <= len(word) <= 40) and word.lower() not in stopwords)
    return accepted


def leaves(tree):
    """Finds leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']):
        yield subtree.leaves()


def normalise(word, stemmer, lem, lem_word_mapping):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word1 = stemmer.stem(word)
    word1 = lem.lemmatize(word1)
    if word != word1:
        lem_word_mapping[word1] = word
    return word1


def get_terms(tree, stopwords, stemmer, lem, lem_word_mapping):
    """Returns the words after checking acceptable conditions, normalizing and lemmatizing"""
    term = [normalise(w, stemmer, lem, lem_word_mapping) for w in tree if acceptable_word(w, stopwords)]
    yield term


def get_t_norm(tree, stopwords, stemmer, lem, lem_word_mapping):
    """Parse leaves in chunk and return after checking acceptable conditions, normalizing and lemmatizing"""
    for leaf in leaves(tree):
        term = [normalise(w, stemmer, lem, lem_word_mapping) for w, t in leaf if acceptable_word(w, stopwords)]
        yield term

def extractOpinionPhrases(posTaggedData, gram, stopwords, stemmer, lem, lem_word_mapping):
    '''Extract noun phrases from part of speech tagged tokenized words'''
    output=[]
    for tup in posTaggedData:
        chunk = nltk.RegexpParser(gram)
        tr = chunk.parse(tup)
        term = get_t_norm(tr, stopwords, stemmer, lem, lem_word_mapping)

        for ter in term:
            wordConcat=""
            for word in ter:
                if wordConcat=="":
                    #Replace good, wonderful and awesome with great
                    wordConcat = wordConcat + word.replace("good","great").replace("wonderful","great").replace("awesome","great").replace("awesom","great")
                else:
                    wordConcat = wordConcat + " " +  word
            if(len(ter)>1):
                output.append(wordConcat)
    return output

def replacewords(mc, lem_word_mapping, p):
    newmc=[]
    for a in mc:
        newword="";found=False;
        for b in a[0].split():
            for x in lem_word_mapping:
                #print(x)
                #print(b)
                if b==x:
                    found=True
                    sing=(lem_word_mapping[x] if p.singular_noun(lem_word_mapping[x])==False else p.singular_noun(lem_word_mapping[x]))
                    if newword=="":
                        newword = newword + sing
                    else:
                        newword = newword + " " +  sing
            if found==False:
                if newword=="":
                    newword = newword + b
                else:
                    newword = newword + " " +  b
                    #print(newword)
        newmc.append((newword,a[1]))
    return newmc


def extract_key_features(positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list):
    pos_tokens, pos_tokens_postagged, neut_tokens, neut_tokens_postagged, neg_tokens, neg_tokens_postagged = tokenize_review_sentences(positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list)


    stopwords = nltk.corpus.stopwords.words('english')
    lem = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    lem_word_mapping = {}

    gram = r"""       
        P1:{<JJ><NN|NNS>}
        P2:{<JJ><NN|NNS><NN|NNS>}
        P3:{<RB|RBR|RBS><JJ>}
        P4:{<RB|RBR|RBS><JJ|RB|RBR|RBS><NN|NNS>}
        P5:{<RB|RBR|RBS><VBN|VBD>}
        P6:{<RB|RBR|RBS><RB|RBR|RBS><JJ>}
        P7:{<VBN|VBD><NN|NNS>}
        P8:{<VBN|VBD><RB|RBR|RBS>}
    """

    ExtractedWords_pos = extractOpinionPhrases(pos_tokens_postagged, gram, stopwords, stemmer, lem, lem_word_mapping)
    ExtractedWords_neg = extractOpinionPhrases(neg_tokens_postagged, gram, stopwords, stemmer, lem, lem_word_mapping)

    freqdist_neg = nltk.FreqDist(word for word in ExtractedWords_neg)
    mc_neg = freqdist_neg.most_common()
    freqdist_pos = nltk.FreqDist(word for word in ExtractedWords_pos)
    mc_pos = freqdist_pos.most_common()

    p = inflect.engine()

    final_neg = replacewords(mc_neg, lem_word_mapping, p)
    final_pos = replacewords(mc_pos, lem_word_mapping, p)

    tmp_positive_review_key_features = final_pos[:15]
    tmp_negative_review_key_features = final_neg[:15]
    positive_review_key_features = []
    negative_review_key_features = []

    for item in tmp_positive_review_key_features:
        positive_review_key_features.append(item[0])

    for item in tmp_negative_review_key_features:
        negative_review_key_features.append(item[0])
    return positive_review_key_features, negative_review_key_features

def main():
    positive_original_review_text_list = []
    neutral_original_review_text_list = []
    negative_original_review_text_list = []
    positive_review_key_features, negative_review_key_features = extract_key_features(positive_original_review_text_list, neutral_original_review_text_list, negative_original_review_text_list)

if __name__ == '__main__':
    main()
