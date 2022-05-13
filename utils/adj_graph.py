from graph import generate_adj_data_from_grounded_concepts__use_LM
from data_utils import load_sparse_adj_data_with_contextnode
from grounding import create_matcher_patterns

if __name__ == "__main__":
    # create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    generate_adj_data_from_grounded_concepts__use_LM("../data/protoqa/grounded/train.grounded.jsonl",
     "../data/cpnet/conceptnet.en.pruned.graph", "../data/cpnet/concept.txt", "../data/protoqa/graph/train_20k.graph.adj.pk", 48, is_dev=False, debug=False)

    # load_sparse_adj_data_with_contextnode("../data/protoqa/train.graph.adj.pk", 200, 768, None)