import os
from data import *
from collections import defaultdict
import numpy as np

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    new_train_data = [val for sublist in train_data for val in sublist]
    my_ids, help = zip(*new_train_data)#[seq[0] for seq in new_train_data]
    my_unique_ids = list(set(my_ids))

    # set default value for dictionary
    most_freq = [(x,'O') for x in my_unique_ids]
    most_freq = dict(most_freq)

    # get most frequent tag for all words
    for this_word in my_unique_ids:
        filter = [this_word]
        tuples_filtered = [(x, y) for (x, y) in new_train_data if x in filter]
        # tuples_filtered = new_train_data.copy()
        help, this_tags = zip(*tuples_filtered)#[seq[1] for seq in tuples_filtered]
        this_counts = [[x, this_tags.count(x)] for x in set(this_tags)]
        largest = sorted(this_counts, key=lambda x: x[1], reverse=True)[0]
        most_freq[this_word] = largest[0]

    return most_freq
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs_help = []
        for word in words:
            try:
                pred_tag_seqs_help.append(pred_tags[word])
            except:
                pred_tag_seqs_help.append('O')
        pred_tag_seqs.append(tuple(pred_tag_seqs_help))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

