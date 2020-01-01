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
    d = extra_decoding_arguments
    #print(d)
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
                #########
                # all_features = []
                # for t in index_to_tag_dict.keys():
                #     if t == num_tags - 1:
                #         continue
                #     curr_word = sent[k][0]
                #     next_word = sent[k + 1][0] if k < (len(sent) - 1) else '</s>'
                #     prev_word = sent[k - 1][0] if k > 0 else '<st>'
                #     prevprev_word = sent[k - 2][0] if k > 1 else '<st>'
                #     prev_tag = index_to_tag_dict[u]
                #     prevprev_tag = index_to_tag_dict[t]
                #     if (curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag) in d:
                #         all_features.append(d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)])
                #     else:
                #         all_features.append(extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag,prevprev_tag))
                #         d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)] = all_features[-1]
                # all_features_vec = vec.fit_transform(all_features)
                # preds_for_v = logreg.predict_proba(all_features_vec)[:, v]
                # pi = pi[k - 1, :, u]
                # products = preds_for_v * pi if k > 0 else preds_for_v

                #########
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
                    # if (curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag) in d:
                    #     features = d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)]
                    # else:
                    #     features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag,prevprev_tag)
                    #     d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)] = features
                    #
                    # # missing the feature for the label t that we are considering!!!
                    # features_vec = vectorize_features(vec, features)
                    # products[t] = logreg.predict_proba(features_vec)[0][v] * pi[k - 1][t][u] if k > 0 else \
                    # logreg.predict_proba(features_vec)[0][v]
###################################
                    # if (curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag) in d:
                    #     pred = d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)]
                    # else:
                    #     features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
                    #     features_vec = vectorize_features(vec, features)
                    #     pred = logreg.predict_proba(features_vec)[0]
                    #     d[(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)] = pred
                    # products[t] = pred[v] * pi[k - 1][t][u] if k > 0 else pred[v]
         ################################3
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
    ### END YOUR CODE
    return predicted_tags