import re
import spacy
from string import punctuation
from collections import defaultdict, OrderedDict
import pymorphy2
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
from prettytable import PrettyTable
dataset=[];
with open("dataset.txt", "r",encoding="utf-8") as file:
    dataset= re.split(r'Документ №', str(file.readlines()))


def PunctuationReplace(s):
    return ''.join(c for c in s if c not in punctuation)

def Tokenizer(inp):
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    return nlp(inp)


#dataset.remove(dataset[24])

docCounter = 1
objs = defaultdict(list)
counter = 1

for inp in dataset:
    inp = inp.lower()
    inp = PunctuationReplace(inp)
    inp = re.sub('[-«»—]', '', inp)
    doc = Tokenizer(inp)
    morph = pymorphy2.MorphAnalyzer()
    for token in list(doc):
        token = token.lemma_
        if token not in objs.keys():
            p = morph.parse(token)[0]
            objs[token].append(counter)
            objs[token].append(p.tag.POS)
            objs[token].append(1)
            objs[token].append(docCounter)
            counter += 1
        else:
            objs[token][2] += 1
    for i in list(objs):
        if re.match('\s', i[0]) is not None and i is not ' ':
            rep = ""
            objs.pop(i[0])

    docCounter += 1

def TableFiller(objs):
    x = PrettyTable()
    x.field_names = ["ID", "Token", "No. of document", "Count", "POS"]
    for i in objs.items():
        Token = i[0]
        ID = i[1][0]
        POS = i[1][1]
        Count = i[1][2]
        NumbDoc = i[1][3]
        x.add_row([ID, Token, NumbDoc, Count, POS])
    return x

print(TableFiller(objs))