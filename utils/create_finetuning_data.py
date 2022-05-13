from distutils.log import debug
import json
from tqdm import tqdm
import numpy as np
import itertools
from collections import defaultdict

from multiprocessing import Pool


from datasets import load_dataset, Features, Value
from elasticsearch import Elasticsearch



def get_finetuning_data_baseline_one_and_two(triple_path, statement_path, output_path, debug=False):
    with open(triple_path, 'r', encoding='utf-8') as fin_triple, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_triples = fin_triple.readlines()
        lines_state = fin_state.readlines()
        questions = {}
        for j, line in enumerate(lines_state):
            dic = json.loads(line)
            questions[dic['metadata']['id']] = dic['question']['normalized']
        if debug:
            lines_triples = lines_triples[0:5]
        gpt_input_strings = []
        question_strings = {}
        for j, line in enumerate(lines_triples):
            dic = json.loads(line)
            for qid in dic:
                answers_and_triples = dic[qid]
                question = questions[qid]
                for answer in answers_and_triples:
                    triples = answers_and_triples[answer]
                    triples = triples[::-1]
                    # take only last 10. i.e, the ones with the high score.
                    triples = triples[-10:]
                    trip_str = ''
                    for trip in triples:
                        trip_str += trip + '. '
                    input_string =  trip_str  + ' '+  question + '\t' + answer
                    # input_string =   question +' '+ trip_str+  '\t' + answer
                    question_strings[qid] =input_string 
                    gpt_input_strings.append(input_string)

        # # Write the input strings to a file
        # with open(output_path, 'w') as fout:
        #     for sentence in gpt_input_strings:
        #         fout.write(sentence)
        #         fout.write('\n')

        # Write the input strings to a file
        with open(output_path, 'w') as fout:
            
            for qid in question_strings:
                new_dic = {}
                new_dic['metadata'] = {
                    'id' : qid
                }
                new_dic['question'] = {}
                new_dic['question']['normalized'] = question_strings[qid]

                json.dump(new_dic, fout)
                fout.write('\n')



def contains_word(s, w):
    return (' ' + s + ' ') in (' ' + w + ' ')

def get_wiki_sentences_for_triple(triple):
    sub, obj = triple.split('___')
    if sub in stopwords or obj in stopwords:
        return []
    sub_filter = dataset.filter(lambda example: contains_word(sub.lower(),  example['text'].lower()) , num_proc =200)
    sub_obj_filter = sub_filter.filter(lambda example: contains_word(obj.lower(),  example['text'].lower()) , num_proc = 100)
    if len(sub_obj_filter):
        retrieved = sub_obj_filter['text']
        retrieved_len  = len(retrieved)
        if retrieved_len > 10:
            retrieved = sub_obj_filter['text'][:10]
        return retrieved
    else:
        return []

def get_finetuning_data_wikipedia(triple_path, statement_path, finetune_data_path, wiki_sentences_path, debug=False, num_processes=48):

    with open(triple_path, 'r', encoding='utf-8') as fin_triple, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_triples = fin_triple.readlines()
        lines_state = fin_state.readlines()
        questions = {}
        for j, line in enumerate(lines_state):
            dic = json.loads(line)
            questions[dic['metadata']['id']] = dic['question']['normalized']
        if debug:
            lines_triples = lines_triples[0:1]
        gpt_input_strings = []
        question_answer_related_wiki_sentences = defaultdict(lambda: defaultdict(list))
        for j, line in enumerate(lines_triples):
            dic = json.loads(line)
            for qid in dic:
                answers_and_triples = dic[qid]
                question = questions[qid]
                for answer in answers_and_triples:
                    triples = answers_and_triples[answer]
                    # Take the top 10 triples
                    triples = triples[:10]
                    sent_str = ''
                    res1 = [get_wiki_sentences_for_triple(triple) for triple in tqdm(triples)]
                    sents = list(itertools.chain(*res1))
                    for sent in sents:
                        question_answer_related_wiki_sentences[qid][answer].append(sent)
                        sent_str += sent +  ' '
                    input_string = sent_str + ' ' + question + '\t' + answer
                    gpt_input_strings.append(input_string)
                    
        # Write the input strings to a file
        with open(finetune_data_path, 'w') as fout:
            for sentence in gpt_input_strings:
                fout.write(sentence)
                fout.write('\n')

        with open(wiki_sentences_path, 'w') as fout:
            for key in question_answer_related_wiki_sentences:
                json.dump({key: question_answer_related_wiki_sentences[key]}, fout)
                fout.write('\n')
                
                    
if __name__ == "__main__":
    get_finetuning_data_baseline_one_and_two("../data/protoqa/triples/sentence.dev.10.jsonl", "../data/protoqa/statement/dev.statement.jsonl", "../data/protoqa/finetuned/triples_finetune_dev_new.txt", debug=False)
    

    # with open("gist_stopwords.txt", "r") as fin:
    #     content = fin.read()
    #     stopwords = content.split(",")
    
    # emotion_features = Features({'text': Value('string')})
    # dataset = load_dataset("text", split="train", data_files="../baseline2/data/wikisent2.txt", features=emotion_features,   cache_dir='../baseline2/cache')
    # get_finetuning_data_wikipedia("../data/protoqa/triples/wikipedia.train.4.jsonl",
    #  "../data/protoqa/statement/train.statement.jsonl", "../data/protoqa/wikipedia/finetune_train.txt", "../data/protoqa/wikipedia/wiki_train_sentences.txt", debug=False, num_processes=48)




