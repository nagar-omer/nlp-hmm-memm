import sys
import os
sys.path.insert(0,  "..")
from MEMM.ConvertFeatures import create_map_and_sparse
from MEMM.ExtractFeatures import create_ftr_file_for
from MEMM.MEMMTag import MEMMTagger
from MEMM.TrainSolver import create_model_file


if __name__ == "__main__":
    args = sys.argv
    if len(sys.argv) < 4:
        print("input\t\tNerHmmTag:\ttrain_file_name,\ttest_file_name,\t out_file_name\n\n")

    train = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "train")
    test = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(__file__.replace("/", os.sep).rsplit(os.sep, 1)[0], "ner_data", "dev")
    out_name = sys.argv[2] if len(sys.argv) > 3 else "ner_eval.memm.pred"


    create_ftr_file_for(train, "ner_ftr_file")
    create_map_and_sparse("ner_ftr_file", "ner_ftr_vec_file", "ner_ftr_map_file")
    create_model_file("ner_ftr_vec_file", "ner_model")
    MEMMTagger(test, "ner_model", "ner_ftr_map_file").memm_tag(out_name=out_name)
