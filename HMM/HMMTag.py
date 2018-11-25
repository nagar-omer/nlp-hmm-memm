import sys
import os
sys.path.insert(0, "..")
from MLETrain import *
from utils.viterbi import ViterbiAlg


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("input\t\tHMMTag:\tinput_file_name,\t q.mle, e.mle,\t out_file_name,\t extra_file_name\n\n")


    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]
    extra_file_name = sys.argv[5] if len(sys.argv) > 5 else "."

    me = MleExtractor(e_mle_filename, q_mle_filename, extra_file_name)

    src_file = open(input_file_name, "rt")  # open file
    out_file = open(out_file_name, "wt")
    tagger = ViterbiAlg(me._pos_list, me.prob_func)
    for line in src_file:
        # ---------- BREAK -----------
        seq = []
        label = []
        for w_p in line.split():  # break line to [.. (word, POS) ..]
            word, pos = w_p.rsplit("/", 1)
            seq.append(word)
            label.append(pos)
        pred = tagger.pred_viterbi(seq, log=True)  # predict
        for i, word in enumerate(seq):
            out_file.write(word + "/" + str(pred[i]) + " ")
        out_file.write("\n")

        # print results
        identical = sum([1 for p, l in zip(pred, label) if p == l])
        recall = str(int(identical / len(pred) * 100))
        print("pred: " + str(pred) + "\nlabel: " + str(label) +
              "\nrecall:\t" + str(identical) + "/" + str(len(pred)) + "\t~" + recall + "%")


