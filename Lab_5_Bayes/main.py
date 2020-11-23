import re
import math
import spacy
from string import punctuation
from collections import defaultdict, OrderedDict
import pymorphy2
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
from prettytable import PrettyTable

# region Database

dataset = [];
documents_Name = {'C1_1', 'C1_2', 'C1_3', 'C2_1', 'C2_2', 'C2_3', 'C3_1', 'C3_2', 'C3_3'};


def import_documents():
    for document in documents_Name:
        dataset.append(get_document_data(document))


def get_document_data(file):
    with open(""+file + ".txt", "r", encoding="utf-8") as file:
        return file.readlines()


# endregion


# region preprocessing
def special_symbol_replace(objs):
    return re.sub("n", ".", objs)


def punctuation_replace(s):
    return ''.join(c for c in s if c not in punctuation)


def tokenizer(inp):
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    return nlp(inp)


def remove_empties(objs):
    for i in list(objs):
        if re.match('\s', i[0]) is not None and i is not ' ' and i[0] in objs.keys():
            objs.pop(i[0])
    return objs
# endregion
# region VSM
def Get_vector_frequency(dataset):
    objs=defaultdict(list)
    id_document = 0
    for inp in dataset:
        inp = inp.lower()
        inp=special_symbol_replace(inp)
        inp = punctuation_replace(inp)
        inp = re.sub('[-«»—]', '', inp)
        doc = tokenizer(inp)
        lengths_of_documents.append(len(doc))
        for token in list(doc):
            token = token.lemma_
            if token not in objs.keys():
                for i in range(0, dataset.__len__(), 1):
                    objs[token].append(0)
                objs[token][id_document] += 1;
            else:
                objs[token][id_document] += 1;
        objs= remove_empties(objs)
        id_document += 1
    return objs

def Boolean(objs):
    for item in objs.items():
        for i in range(2,26,1):
            if item[1][i]>1:
                item[1][i]=1;
    return objs;

def IDF(objs):
    idf=[]
    for item in objs.items():
        count=0;
        for i in range(2, 26, 1):
            if item[1][i] >= 1:
                count+=1;
        idf.append(math.log(len(dataset)/count))
    return idf

def TF_IDF(objs,lenght):
    idf=IDF(objs)
    index_idf = 0
    for item in objs.items():
        for j in range(2,26,1):
            item[1][j]=item[1][j]/lenght[j-2]*idf[index_idf]
        index_idf+=1;
    return objs
# endregion
lengths_of_documents=[];
vectors = Get_vector_frequency(dataset);


def table_filler(objs):
    x = PrettyTable()
    x.field_names = ["Token",'C1_1', 'C1_2', 'C1_3', 'C2_1', 'C2_2', 'C2_3', 'C3_1', 'C3_2', 'C3_3']
    objs =  OrderedDict(sorted(objs.items(), key=lambda item: item[1][1], reverse=True))
    for i in objs.items():
        x.add_row(["Текст " + str(i[1][0]), i[1][1], i[1][2], i[1][3]])
    return x


print(table_filler(vectors))
print('fdgds');