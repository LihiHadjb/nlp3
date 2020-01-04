from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict



def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    d = extra_decoding_arguments
    if sent in d:
        return d[sent]

    num_tags = len(index_to_tag_dict)
    pi = np.ones((len(sent), num_tags - 1, num_tags - 1))
    bp = np.zeros((len(sent), num_tags - 1, num_tags - 1), dtype=int)
    for k in range(len(sent)):
        for u in index_to_tag_dict.keys():
            if u == num_tags - 1:
                continue
            for v in index_to_tag_dict.keys():
                if v == num_tags - 1:
                    continue
                products = np.empty(num_tags - 1)
                for t in index_to_tag_dict.keys():
                    if t == num_tags - 1:
                        continue
                    curr_word = sent[k][0]
                    next_word = sent[k + 1][0] if k < (len(sent) - 1) else '</s>'
                    prev_word = sent[k - 1][0] if k > 0 else '<st>'
                    prevprev_word = sent[k - 2][0] if k > 1 else '<st>'
                    prev_tag = index_to_tag_dict[u]
                    prevprev_tag = index_to_tag_dict[t]
                    features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
                    features_vec = vectorize_features(vec, features)
                    products[t] = logreg.predict_proba(features_vec)[0][v] * pi[k - 1][t][u] if k > 0 else logreg.predict_proba(features_vec)[0][v]
                best_index = np.argmax(products)
                bp[k][u][v] = best_index
                pi[k][u][v] = products[best_index]

    if len(sent) == 1:
        predicted_tags[0] = best_index
    else:
        predicted_tags[-2], predicted_tags[-1] = np.unravel_index(pi[-1].argmax(), pi[-1].shape)
        for i in range(len(sent) - 3, -1, -1):
            predicted_tags[i] = bp[i + 2][predicted_tags[i + 1]][predicted_tags[i + 2]]

    print(predicted_tags)
    predicted_tags = [index_to_tag_dict[index] for index in predicted_tags]
    d[sent] = predicted_tags
    ### END YOUR CODE
    return predicted_tags






####old version
def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    d_preds = extra_decoding_arguments[0]
    d_sents = extra_decoding_arguments[1]
    h = tuple(sent)
    if h in d_sents:
        return d_sents[h]

    num_tags = len(logreg.classes_)
    prev_pi = np.ones((num_tags, num_tags))
    bp = np.zeros((len(sent), num_tags, num_tags), dtype=int)

    for k in range(len(sent)):
        for u in logreg.classes_:
            for v in logreg.classes_:
                curr_word = sent[k]
                next_word = sent[k + 1] if k < (len(sent) - 1) else '</s>'
                prev_word = sent[k - 1] if k > 0 else '<st>'
                prevprev_word = sent[k - 2] if k > 1 else '<st>'
                prev_tag = index_to_tag_dict[u]
                for t in logreg.classes_:
                    prevprev_tag = index_to_tag_dict[t]

    ### END YOUR CODE
    return predicted_tags






















def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    d_preds = extra_decoding_arguments[0]
    d_sents = extra_decoding_arguments[1]
    num_tags = len(logreg.classes_)


