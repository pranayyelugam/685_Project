import collections
import glob
import json
import random
import stanza
import tqdm
import numpy as np
random.seed(131)

with open("../data/nltk_stopwords.json", "r") as f:
  STOPWORDS = json.load(f)

STANZA_PIPELINE = stanza.Pipeline("en",
                                processors="tokenize,lemma",
                                tokenize_no_ssplit=True)

for filename in glob.glob('../data/wikidata/*'):
    print(filename)
    file = open(filename, "r")
    doclist = [ line for line in file ]

truncated_doclist  = doclist[:1000]

f = open('lemmatized_docs_sbatch_version', 'w')

def stanza_preprocess(sentences):
    batch_len = 1000
    k = int(len(sentences) / batch_len)
    lemmatized = []
    for i in tqdm.tqdm(range(0, k)):
        l = i *batch_len
        batched_sent = doclist[l:l+batch_len]
        doc = STANZA_PIPELINE("\n\n".join(batched_sent))
        for sentence in doc.sentences:
            sentence_lemmas = []
            for token in sentence.tokens:
                (token_dict,) = token.to_dict()
                if "lemma" in token_dict:
                    maybe_lemma = token_dict["lemma"].lower()
                    if maybe_lemma not in STOPWORDS:
                        sentence_lemmas.append(maybe_lemma)
                        f.write(maybe_lemma)
                        f.write('\n')
            lemmatized.append(sentence_lemmas)
    return lemmatized


lemmatized = stanza_preprocess([x for x in (truncated_doclist)])
f.close()