from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict


def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE

    extra_decoding_arguments = []
    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    n = len(curr_word)
    features['tag trigram'] = prevprev_tag + "_" + prev_tag
    features['tag bigram'] = prev_tag
    # features['tag unigram'] =
    features['prev tag'] = prev_word + "_" + prev_tag
    features['prevprev tag'] = prevprev_word + "_" + prevprev_tag

    features['pre1'] = curr_word[0]
    features['suff1'] = curr_word[-1]

    if n >= 2:
        features['pre2'] = curr_word[0:2]
        features['suff2'] = curr_word[n - 2:n]

    if n >= 3:
        features['pre3'] = curr_word[0:3]
        features['suff3'] = curr_word[n - 3:n]

    if n >= 4:
        features['pre4'] = curr_word[0:4]
        features['suff4'] = curr_word[n - 4:n]

    if n >= 5:
        features['pre5'] = curr_word[0:5]
        features['suff5'] = curr_word[n - 5:n]

    # features['pre2'] = curr_word[0:2] if n >= 2 else ""
    # features['pre3'] = curr_word[0:3] if n >= 3 else ""
    # features['pre4'] = curr_word[0:4] if n >= 4 else ""
    # features['pre5'] = curr_word[0:5] if n >= 5 else ""
    #
    # features['suff1'] = curr_word[-1]
    # features['suff2'] = curr_word[n-2:n] if n >= 2 else ""
    # features['suff3'] = curr_word[n-3:n] if n >= 3 else ""
    # features['suff4'] = curr_word[n-4:n] if n >= 4 else ""
    # features['suff5'] = curr_word[n-5:n] if n >= 5 else ""
    ### END YOUR CODE
    return features


def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1],
                                 prevprev_token[1])


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    for i in range(len(sent)):
        curr_word = sent[i][0]
        next_word = sent[i + 1][0] if i < (len(sent) - 1) else '</s>'
        prev_word = sent[i - 1][0] if i > 0 else '<st>'
        prevprev_word = sent[i - 2][0] if i > 1 else '<st>'
        prev_tag = predicted_tags[i - 1] if i > 0 else '*'
        prevprev_tag = predicted_tags[i - 2] if i > 1 else '*'

        features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
        features_vec = vectorize_features(vec, features)
        best_index = logreg.predict(features_vec)[0]
        best_tag = index_to_tag_dict[best_index]
        predicted_tags[i] = best_tag

    ### END YOUR CODE
    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    print("-------------------------------------------------------------------------------------------------------------")
    num_tags = len(index_to_tag_dict)
    pi = np.ones((len(sent), num_tags-1, num_tags-1))
    bp = np.zeros((len(sent), num_tags-1, num_tags-1), dtype=int)
    for k in range(len(sent)):
        #print(pi[k - 1])
        for u in index_to_tag_dict.keys():
            if u == num_tags-1:
                continue
            for v in index_to_tag_dict.keys():
                if v == num_tags - 1:
                    continue
                products = np.empty(num_tags-1)
                for t in index_to_tag_dict.keys():
                    if t == num_tags-1 :
                        continue
                    curr_word = sent[k][0]
                    next_word = sent[k + 1][0] if k < (len(sent) - 1) else '</s>'
                    prev_word = sent[k - 1][0] if k > 0 else '<st>'
                    prevprev_word = sent[k - 2][0] if k > 1 else '<st>'
                    prev_tag = index_to_tag_dict[u]
                    prevprev_tag = index_to_tag_dict[t]
                    features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
                   #missing the feature for the label t that we are considering!!!
                    features_vec = vectorize_features(vec, features)
                    #print(logreg.predict_proba(features_vec)[0][v],pi[k-1][t][u]  )
                    products[t] = logreg.predict_proba(features_vec)[0][v] * pi[k-1][t][u] if k > 0 else logreg.predict_proba(features_vec)[0][v]
                best_index = np.argmax(products)
                #print(products)
                bp[k][u][v] = best_index
                pi[k][u][v] = products[best_index]
                #print(products[best_index])
    if len(sent) == 1:
        predicted_tags[0] = np.argmax(pi[-1])
    else:
        predicted_tags[-2], predicted_tags[-1] = np.unravel_index(pi[-1].argmax(), pi[-1].shape)
        for i in range(len(sent)-3, -1, -1):
            predicted_tags[i] = bp[i+2][predicted_tags[i+1]][predicted_tags[i+2]]
    print(predicted_tags)
    predicted_tags = [index_to_tag_dict[index] for index in predicted_tags]
    ### END YOUR CODE
    return predicted_tags


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        greedy_pred_tag_seqs.append(memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        greedy_pred_tag_seqs.append(memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        ### END YOUR CODE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation


def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)

    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
