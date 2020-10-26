import re
import math
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
    dataset.remove(dataset[24])


def special_symbol_Replace(objs):
    return re.sub("n",".",objs)

def Punctuation_Replace(s):
    return ''.join(c for c in s if c not in punctuation)

def Tokenizer(inp):
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')

    return nlp(inp)


def Get_frequency():
    objs=defaultdict(list)
    counter = 1
    docCounter = 1
    for inp in dataset:
        inp = inp.lower()
        inp=special_symbol_Replace(inp)
        inp = Punctuation_Replace(inp)
        inp = re.sub('[-«»—]', '', inp)
        doc = Tokenizer(inp)
        lengths_of_documents.append(len(doc))
        for token in list(doc):
            token = token.lemma_
            if token not in objs.keys():
                objs[token].append(counter)
                objs[token].append(1)
                for i in range(0, 24, 1):
                    objs[token].append(0)
                objs[token][docCounter + 1] += 1;
                counter += 1
            else:
                objs[token][1] += 1
                objs[token][docCounter + 1] += 1;
        for i in list(objs):
            if re.match('\s', i[0]) is not None and i is not ' ' and i[0] in objs.keys():
                objs.pop(i[0])
        docCounter += 1
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



lengths_of_documents=[];
vectors = Get_frequency();
print("Выберите тип веса (1/2/3)")
method=input()
if method=="1":
    vectors=Boolean(vectors);
elif method=="3":
    vectors=TF_IDF(vectors,lengths_of_documents);





def TableFiller(objs):
    x = PrettyTable()
    x.field_names = ["ID", "Token", "Count", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9",
                     "doc10", "doc11", "doc12", "doc13", "doc14", "doc15", "doc16", "doc17", "doc18", "do19", "doc20",
                     "doc21", "doc22", "doc23", "doc24"]
    for i in objs.items():
        Token = i[0]
        ID = i[1][0]
        Count = i[1][1]
        doc1=i[1][2]
        doc2 = i[1][3]
        doc3 = i[1][4]
        doc4 = i[1][5]
        doc5 = i[1][6]
        doc6 = i[1][7]
        doc7 = i[1][8]
        doc8 = i[1][9]
        doc9 = i[1][10]
        doc10 = i[1][11]
        doc11 = i[1][12]
        doc12 = i[1][13]
        doc13 = i[1][14]
        doc14 = i[1][15]
        doc15 = i[1][16]
        doc16 = i[1][17]
        doc17 = i[1][18]
        doc18 = i[1][19]
        doc19 = i[1][20]
        doc20 = i[1][21]
        doc21 = i[1][22]
        doc22 = i[1][23]
        doc23 = i[1][24]
        doc24 = i[1][25]
        x.add_row([ID, Token, Count, doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10, doc11, doc12,
                   doc13, doc14, doc15, doc16, doc17, doc18, doc19, doc20, doc21, doc22, doc23, doc24])
    return x

print(TableFiller(vectors))