# 685_Project

1. Download ConceptNet and extract it in a data folder.
2. _python preprocess.py --run protoqa_ (this creates pickle file for the subgraphs for all the question answer contexts)
3. Execute _python utils/testing.py_ with the pickled graph file to prune and generate the triples for each question answer context.
4. Execute _python utils/create_finetuning_data.py_ to create finetuning data in two different orderings: 1. Triples, Question, Answer and 2. Question, Triples and Answer.
5. Fine-tune GPT2 using this data.
6. The code for BM25 sentence extraction is in the folder bm25.

   
