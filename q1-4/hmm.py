import os
import time
from data import *
from collections import defaultdict, Counter
import numpy as np


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: defaultdict(int))
    ### YOUR CODE HERE
    for sent in sents:
        sent.append(("STOP", "STOP"))
        total_tokens += len(sent)
        sent.insert(0, ("*", "*"))
        sent.insert(0, ("*", "*"))
        for i in range(1, len(sent)):
            uni = sent[i][1]
            if uni not in q_uni_counts:
                q_uni_counts[uni] = 0
            q_uni_counts[uni] += 1

            bi = (sent[i - 1][1], sent[i][1])
            if bi not in q_bi_counts:
                q_bi_counts[bi] = 0
            q_bi_counts[bi] += 1

            if i >= 2:
                tri = (sent[i - 2][1], sent[i - 1][1], sent[i])
                if tri not in q_tri_counts:
                    q_tri_counts[tri] = 0
                q_tri_counts[tri] += 1

                word_tag = sent[i]
                if word_tag not in e_word_tag_counts:
                    e_word_tag_counts[word_tag] = 0
                e_word_tag_counts[word_tag] += 1

    e_tag_counts = dict(q_uni_counts)
    del e_tag_counts[("*", "*")[1]]
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    ##
    # tag pruning first
    factor = 0.03
    words = {}
    for word, tag in e_word_tag_counts.keys():
        if word not in words:
            words[word] = []
        word_pos = (word, tag)
        if word_pos not in e_word_tag_counts:
            words[word].append(0.0)
        else:
            words[word].append(float(e_word_tag_counts[word_pos]) / e_tag_counts[tag])

    max_prob_word = {}
    for word in words:
        max_prob_word[word] = max(words[word])

    e_word_tag_counts_pruned = {}
    for word, tag in e_word_tag_counts.keys():
        word_pos = (word, tag)
        if word_pos not in e_word_tag_counts:
            e_prob = 0.0
        else:
            e_prob = float(e_word_tag_counts[word_pos]) / e_tag_counts[tag]
        if e_prob >= factor * max_prob_word[word]:
            e_word_tag_counts_pruned[(word, tag)] = e_word_tag_counts[(word, tag)]
    ###

    tags_available = e_tag_counts.keys()
    s = len(tags_available)
    n = len(sent)
    # Preparing the table, it starts with zeros for pi(0,*,*) and for entries not filled because of prunning policy
    table = np.zeros((n + 1, s + 1, s + 1), np.float64)
    table[0][s][s] = 1.0
    bp = np.zeros((n + 1, s + 1, s + 1), np.int8)

    # get all table values
    # iterate over k
    for word_ixx, (word, _) in enumerate(sent):
        k = word_ixx + 1
        # iterate over v in S_k
        for v_ixx, v in enumerate(tags_available):
            word_pos = (word, v)
            if word_pos not in e_word_tag_counts_pruned:
                continue
            else:
                e = float(e_word_tag_counts_pruned[word_pos]) / e_tag_counts[v]
            #iterate over u in S_k-1
            if k > 2 or k == 2:
                this_possible_tags = tags_available
            else:
                this_possible_tags = ["*"]
            for u_ixx, u in enumerate(this_possible_tags):
                if u == "*":
                    u_ixx = s
                max_val, max_bp = 0, 0
                if k > 2 or (k == 2 and False):
                    this_possible_tags = tags_available
                else:
                    this_possible_tags = ["*"]
                for w_ixx, w in enumerate(this_possible_tags):
                    if w == "*":
                        w_ixx = s
                    # get weighted q value according to lambda
                    q = 0.0
                    trigram = (w, u, v)
                    if trigram in q_tri_counts:
                        q += (lambda1 * q_tri_counts[trigram]) / q_bi_counts[(w, u)]
                    bigram = (u, v)
                    if bigram in q_bi_counts:
                        q += (lambda2 * q_bi_counts[bigram]) / q_uni_counts[u]
                    if v in q_uni_counts:
                        q += ((1-lambda1-lambda2) * q_uni_counts[v]) / total_tokens

                    val = table[k - 1][w_ixx][u_ixx] * q * e
                    if val > max_val:
                        max_val = val
                        max_bp = w_ixx
                # if max_val == 0: print "prob val"
                table[k][u_ixx][v_ixx] = max_val
                bp[k][u_ixx][v_ixx] = max_bp

    # find optimal assignment for y_n, y_(n-1) according to table
    max_val, max_u, max_v = 0.0, 0, 0
    for u_ixx, u in enumerate(tags_available):
        for v_ixx, v in enumerate(tags_available):
            curr_q = 0.0
            trigram = (u, v, "STOP")
            (w, u, v)
            if trigram in q_tri_counts:
                curr_q += (lambda1 * q_tri_counts[trigram]) / q_bi_counts[(u, v)]
            bigram = (v, "STOP")
            if bigram in q_bi_counts:
                curr_q += (lambda2 * q_bi_counts[bigram]) / q_uni_counts[v]
            if "STOP" in q_uni_counts:
                curr_q += ((1 - lambda1 - lambda2) * q_uni_counts["STOP"]) / total_tokens

            curr_val = table[n][u_ixx][v_ixx] * curr_q
            if curr_val > max_val:
                max_val = curr_val
                max_u, max_v = u_ixx, v_ixx
    predicted_tags[n - 1], y_k_2 = list(tags_available)[max_v], max_v
    predicted_tags[n - 2], y_k_1 = list(tags_available)[max_u], max_u

    # optimal assignment iteratively for all other words
    for k in range(n - 2, 0, -1):
        predicted_tag_index = bp[k + 2][y_k_1][y_k_2]
        predicted_tags[k - 1] = list(tags_available)[predicted_tag_index]
        y_k_2 = y_k_1
        y_k_1 = predicted_tag_index

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        lambda1 = 0.05
        lambda2 = 0.9
        pred = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts,
                                 q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1, lambda2)
        pred_tag_seqs.append(pred)
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    token_cm, (p, r, f1) = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                     e_word_tag_counts, e_tag_counts)
    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
