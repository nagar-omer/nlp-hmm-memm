import sys
import os
sys.path.insert(0, "..")
from HMM.MLETrain import MleEstimator, MleExtractor
from utils.viterbi import ViterbiAlg


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("input\t\tNerHmmTag:\ttrain_file_name,\ttest_file_name,\t out_file_name\n\n")

    train = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "train")
    test = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "test.blind")
    out_name = sys.argv[2] if len(sys.argv) > 3 else "ner.hmm.pred"
    mm = MleEstimator(train)
    mm.mle_count_to_txt("ner_e.mle", "ner_q.mle")

    q_mle_filename = "ner_q.mle"
    e_mle_filename = "ner_e.mle"
    out_file_name = out_name

    me = MleExtractor(e_mle_filename, q_mle_filename, ".")

    src_file = open(test, "rt")  # open file
    out_file = open(out_name, "wt")
    tagger = ViterbiAlg(me._pos_list, me.prob_func)
    for line in src_file:
        # ---------- BREAK -----------
        seq = []
        label = []
        for word in line.split():  # break line to [.. (word, POS) ..]
            seq.append(word)
        pred = tagger.pred_viterbi(seq, log=True)  # predict
        for i, word in enumerate(seq):
            out_file.write(word + "/" + str(pred[i]) + " ")
        out_file.write("\n")
