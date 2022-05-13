# conda env: bm25

import collections
import glob
import json
from rank_bm25 import BM25Okapi
import random
import stanza
from tqdm import tqdm
import numpy as np
random.seed(131)
from datetime import datetime
from collections import defaultdict
import pickle

now = datetime.now()

current_time = now.strftime("%H:%M:%S")





def stanza_preprocess(sentences):
    with open("./data/nltk_stopwords.json", "r") as f:
        STOPWORDS = json.load(f)

    STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize,lemma",
                                  tokenize_no_ssplit=True)

    batch_len = 1000
    f = open('lemmatized_docs_finals_{}.txt'.format(current_time), 'w')
    k = int(len(sentences) / batch_len)
    lemmatized = []
    for i in tqdm(range(0, k)):
        l = i * batch_len
        batched_sent = sentences[l:l+batch_len]
        doc = STANZA_PIPELINE("\n\n".join(batched_sent))
        for sentence in doc.sentences:
            sentence_lemmas = []
            for token in sentence.tokens:
                (token_dict,) = token.to_dict()
                if "lemma" in token_dict:
                    maybe_lemma = token_dict["lemma"].lower()
                    if maybe_lemma not in STOPWORDS:
                        sentence_lemmas.append(maybe_lemma)
                        
            lemmatized.append(sentence_lemmas)
            f.write(str(sentence_lemmas))
            f.write('\n')
    
    f.close()
    return lemmatized


def create_bm25_model(num_sent):
    for filename in glob.glob('./data/wikidata/*'):
        print(filename)
        file = open(filename, "r")
        doclist = [ line for line in file ]

    truncated_list = doclist[:num_sent]
    lemmed = stanza_preprocess([x for x in (truncated_list)])
    model = BM25Okapi(lemmed)

    #To save bm25 object
    with open('bm25result', 'wb') as bm25result_file:
        pickle.dump(model, bm25result_file)

    print("done dumping picklee fie")

def get_corpus(num_sent):
    for filename in glob.glob('./data/wikidata/*'):
        print(filename)
        file = open(filename, "r")
        doclist = [ line for line in file ]
    truncated_list = doclist[:num_sent]
    return truncated_list



def get_finetuning_data_wikipedia(bm25model, triple_path, statement_path, finetune_data_path, wiki_sentences_path, debug=False, num_processes=48):
    with open(triple_path, 'r', encoding='utf-8') as fin_triple, open(statement_path, 'r', encoding='utf-8') as fin_state, open(finetune_data_path, 'w') as fout:
        lines_triples = fin_triple.readlines()
        lines_state = fin_state.readlines()
        questions = {}
        for j, line in enumerate(lines_state):
            dic = json.loads(line)
            questions[dic['metadata']['id']] = dic['question']['normalized']
        if debug:
            lines_triples = lines_triples[0:1]
        gpt_input_strings = []
        corpus = create_bm25_model(1000000)
        question_answer_related_wiki_sentences = defaultdict(lambda: defaultdict(list))
        for  line in tqdm(lines_triples):
            dic = json.loads(line)
            for qid in dic:
                answers_and_triples = dic[qid]
                question = questions[qid]
                for answer in answers_and_triples:
                    triples = answers_and_triples[answer]
                    # Take the top 10 triples
                    triples = triples[:10]
                    for triple in triples:
                        triple_query = triple.split('__')
                        for ans in answer.split(' '):
                            triple_query.append(ans)
                        result_sent = bm25model.get_top_n(triple_query, corpus, n=1)
                        if len(result_sent) > 0:
                            input_string = result_sent[0].strip() + ' ' + question + '\t' + answer
                        else:
                            input_string =  question + '\t' + answer
                        print(input_string)
                        fout.write(input_string)
                        fout.write('\n')
                        gpt_input_strings.append(input_string)
        return gpt_input_strings
                    

def load_model_from_lemmatized(filename):
    with open(filename) as f:
        data = f.readlines()
    terms = []
    for line in data:
        terms.append(line)
    model = BM25Okapi(terms)
    print("Loaded model from file")
    create_bm25_model(num_sent)

    return model

def extract_wikipedia_sentences(model, out_file, debug=False):
    gpt_strings = get_finetuning_data_wikipedia(model, "../data/protoqa/triples/wikipedia.train.10.jsonl",
      "../data/protoqa/statement/train.statement.jsonl", out_file , "../data/protoqa/wikipedia/wiki_train_sentences.txt", debug=debug, num_processes=48)



def get_finetuning_data_wikipedia_11(bm25model, triple_path, statement_path, finetune_data_path, wiki_sentences_path, debug=False, num_processes=48):
    with open(triple_path, 'r', encoding='utf-8') as fin_triple, open(statement_path, 'r', encoding='utf-8') as fin_state, open(finetune_data_path, 'w') as fout:
        lines_triples = fin_triple.readlines()
        lines_state = fin_state.readlines()
        questions = {}
        for j, line in enumerate(lines_state):
            dic = json.loads(line)
            questions[dic['metadata']['id']] = dic['question']['normalized']
        if debug:
            lines_triples = lines_triples[0:1]
        gpt_input_strings = []
        corpus = get_corpus(2000000)
        question_answer_related_wiki_sentences = defaultdict(lambda: defaultdict(list))
        answer_set = set()
        for  line in tqdm(lines_triples):
            dic = json.loads(line)
            for qid in dic:
                answers_and_triples = dic[qid]
                question = questions[qid]
                for answer in answers_and_triples:
                    triples = answers_and_triples[answer]
                    # Take the top 10 triples
                    triples = triples[:10]
                    for triple in triples:
                        
                        triple_query = triple.split('__')
                        for ans in answer.split(' '):
                            triple_query.append(ans)
                        result_sent = bm25model.get_top_n(triple_query, corpus, n=1)
                        for ans in result_sent:
                            if ans not in answer_set:
                                fout.write(ans)
                                fout.write('\n')
                                answer_set.add(ans)
                    
                    

if __name__ == "__main__":
    # model = create_bm25_model(2000000)
    # # extract_wikipedia_sentences(model, "../data/protoqa/wikipedia/finetune_train_10K_questions_1_bm25_{}.txt".format(current_time), 
    # # debug =False)
    with open('bm25result', 'rb') as bm25result_file:
        model = pickle.load(bm25result_file)
    out_file ="../data/protoqa/wikipedia/finetune_train_10K_questions_1_bm25_{}.txt".format(current_time)
    print(out_file)
    get_finetuning_data_wikipedia_11(model, "../data/protoqa/triples/wikipedia.train.10.jsonl",
      "../data/protoqa/statement/train.statement.jsonl", out_file , "../data/protoqa/wikipedia/wiki_train_sentences.txt", debug=False, num_processes=48)


    


