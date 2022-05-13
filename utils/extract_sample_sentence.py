from distutils.util import convert_path
from unicodedata import name
import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import re

try:
    from .utils import check_file
except ImportError:
    from utils import check_file

__all__ = ['extract_english', 'construct_graph', 'merged_relations']

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]


def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

def extract_english():
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting English concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    print(num_lines)
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):

            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                if "surfaceText" in data:
                    sentence = data["surfaceText"].strip()
                    sentence = re.sub(r"[\([{})\]]", "", sentence)
                else:
                    sentence = ""

                print('\t'.join([rel, head, tail, str(data["weight"]), sentence]) + '\n')

                # fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                # for w in [head, tail]:
                #     if w not in concepts_seen:
                #         concepts_seen.add(w)
                #         cpnet_vocab.append(w)

    # with open(output_vocab_path, 'w') as fout:
    #     for word in cpnet_vocab:
    #         fout.write(word + '\n')

    # print(f'extracted ConceptNet csv file saved to {output_csv_path}')
    # print(f'extracted concept vocabulary saved to {output_vocab_path}')
    # print()

conceptnet_path = './data/cpnet/conceptnet-assertions-5.6.0.csv'
output_csv_path = './data/protoqa/conceptnet.en.csv'

if __name__ == "__main__":
    extract_english()
