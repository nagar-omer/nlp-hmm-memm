import sys
import os
sys.path.insert(0, "..")
from HMM.MLETrain import MleEstimator, MleExtractor
from utils.viterbi import ViterbiAlg
from utils.ner_eval import ner_eval


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("input\t\tNerHmmTag:\ttrain_file_name,\ttest_file_name,\t out_file_name\n\n")

    train = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "train")
    test = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "dev")
    pred_file_name = "ner_eval.hmm.pred"
    mm = MleEstimator(train)
    mm.mle_count_to_txt("ner_e.mle", "ner_q.mle")

    q_mle_filename = "ner_q.mle"
    e_mle_filename = "ner_e.mle"
    out_file_name = pred_file_name

    me = MleExtractor(e_mle_filename, q_mle_filename, ".")

    src_file = open(test, "rt")  # open file
    pred_file = open(pred_file_name, "wt")
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
            pred_file.write(word + "/" + str(pred[i]) + " ")
        pred_file.write("\n")
    ner_eval(test, pred_file_name)
