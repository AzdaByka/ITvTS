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
with open("../Lab 3 create_vector/dataset.txt", "r",encoding="utf-8") as file:
    dataset= re.split(r'Документ №', str(file.readlines()))
    dataset.remove(dataset[24])

# region preprocessing
def special_symbol_Replace(objs):
    return re.sub("n",".",objs)

def Punctuation_Replace(s):
    return ''.join(c for c in s if c not in punctuation)

def Tokenizer(inp):
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    return nlp(inp)

def Remove_empties(objs):
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
        inp=special_symbol_Replace(inp)
        inp = Punctuation_Replace(inp)
        inp = re.sub('[-«»—]', '', inp)
        doc = Tokenizer(inp)
        lengths_of_documents.append(len(doc))
        for token in list(doc):
            token = token.lemma_
            if token not in objs.keys():
                for i in range(0, dataset.__len__(), 1):
                    objs[token].append(0)
                objs[token][id_document] += 1;                
            else:
                objs[token][id_document] += 1;
        objs= Remove_empties(objs)
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
# region Measures


def Calculate_Measures(query, TFModel, counter):
    measures = defaultdict(list)
    for i in range(counter):
        cur = defaultdict(list)
        for key, value in TFModel.items():
            cur[key].append(value[i])
        cur = DictAdd(query, cur)
        query = DictAdd(cur, query)
        measures[i].append(CalcCos(query, cur))
        measures[i].append(CalcJacc(query, cur))
        measures[i].append(CalcDice(query, cur))
    return measures


def CalcCos(query, cur):
    summ = 0
    sumsqrt1 = 0
    sumsqrt2 = 0
    for key,val in query.items():
        summ += cur[key][0] * query[key][0]
        sumsqrt1 += cur[key][0] ** 2
        sumsqrt2 += query[key][0] ** 2
    sumsqrt1 = math.sqrt(sumsqrt1)
    sumsqrt2 = math.sqrt(sumsqrt2)
    res = summ / (sumsqrt1 * sumsqrt2)
    return res


def CalcJacc(query, cur):
    numerator = 0
    denumerator = 0
    for key, val in query.items():
        numerator += min(query[key][0], cur[key][0])
        denumerator += max(query[key][0], cur[key][0])
    res = numerator / denumerator
    return res

def CalcDice(query, cur):
    numerator = 0
    denumerator = 0
    for key, val in query.items():
        numerator += min(query[key][0], cur[key][0])
        denumerator += query[key][0] + cur[key][0]
    res = (2 * numerator) / denumerator
    return res

def DictAdd(query, cur):
    for key, val in query.items():
        if key not in cur:
            cur[key].append(0)
    return cur

# endregion

print("Введите номера документов. Для окончания ввода введите s")
inputDocuments = []
inpNumber = []
while True:
    inp = str(input())
    if inp == "s":
        break
    else:
        inpNumber.append(int(inp))
        inputDocuments.append(dataset[int(inp)])
lengths_of_documents=[];
vectors = Get_vector_frequency(inputDocuments);
query = dataset[1];
vector_query=Get_vector_frequency([query])
measures=Calculate_Measures(vector_query,vectors,inputDocuments.__len__())

def TableFiller(objs):
    x = PrettyTable()
    x.field_names = ["Текст", "Cosine", "Jaccard", "Dice"]
    for i in objs.items():
        Token = i[0]
        ID = i[1][0]
        Count = i[1][1]
        doc1=i[1][2]
        doc2 = i[1][3]
        doc3 = i[1][4]
        doc4 = i[1][5]

        x.add_row([ID, Token, Count, doc1, doc2, doc3, doc4])
    return x

print(TableFiller(vectors))