
from data_utils import *

if __name__ == "__main__":
    # create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    # generate_adj_data_from_grounded_concepts__use_LM("./ground_res.jsonl", "../data/cpnet/conceptnet.en.pruned.graph", "../data/cpnet/concept.txt", "../data/protoqa/train.graph.adj.pk", 48)
    train_adj_path = "../data/protoqa/graph/dev.onlyq.graph.adj.pk"
    # 
    # load_protoqa_sparse_adj_data_with_contextnode   load_sparse_adj_data_with_contextnode
    *train_decoder_data, train_adj_data = load_protoqa_sparse_adj_data_with_contextnode(train_adj_path, 200, 768, None, baseline="sentence")