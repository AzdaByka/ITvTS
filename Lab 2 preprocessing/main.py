import spacy
from string import punctuation
from collections import defaultdict, OrderedDict
import pymorphy2
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
from prettytable import PrettyTable
import re
import csv


def FillDict():
    dict = defaultdict(list)
    dict["N"].append("NOUN")
    dict["A"].append("ADJF")
    dict["A"].append("ADJS")
    dict["A"].append("PRTF")
    dict["A"].append("PRTS")
    dict["A"].append("NUMR")
    dict["A"].append("NPRO")
    dict["V"].append("VERB")
    dict["V"].append("INFN")
    dict["V"].append("GRND")
    dict["D"].append("ADVB")
    dict["P"].append("PREP")
    dict["P"].append("PRCL")
    return dict


def UtfCod(s):
    return s


def ToLowerCase(s):
    return s.lower()


def PunctuationReplace(s):
    return ''.join(c for c in s if c not in punctuation)


def Tokenizer(inp):
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    return nlp(inp)


dict = FillDict()


def PosChanger(inp):
    res = ""
    for i in dict.items():
        if inp in i[1]:
            res = i[0]
    if res == "":
        res = inp
    return res


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


def CsvImport(file_obj):
    file_reader = csv.reader(r_file, delimiter=";")
    documents = []
    count = 0
    for row in file_reader:
        if not count < 2:
            documents.append(row[4])
        count += 1
    return documents



with open("aaa.csv") as r_file:
    documents = CsvImport(r_file)
print("Введите номера документов через enter. Для окончания ввода введите stop")
inputDocuments = []
while True:
    inp = str(input())
    if inp == "stop":
        break
    else:
        inputDocuments.append(documents[int(inp)])
print("Выберите тип сортировки: Сортировка по алфавиту / Сортировка по длине слова (1 / 2)")
sortType = str(input())
docCounter = 1
objs = defaultdict(list)
counter = 1
for inp in inputDocuments:
    inp = UtfCod(inp)
    inp = ToLowerCase(inp)
    inp = PunctuationReplace(inp)
    inp = re.sub('[-«»—]', '', inp)
    doc = Tokenizer(inp)
    morph = pymorphy2.MorphAnalyzer()
    for token in list(doc):
        token = token.lemma_
        if token not in objs.keys():
            p = morph.parse(token)[0]
            objs[token].append(counter)
            objs[token].append(PosChanger(p.tag.POS))
            objs[token].append(1)
            objs[token].append(docCounter)
            counter += 1
        else:
            objs[token][2] += 1
    for i in list(objs):
        if re.match('\s', i[0]) is not None:
            rep = ""
            objs.pop(i[0])

    docCounter += 1
if sortType is "1":
    objs = OrderedDict(sorted(objs.items()))
else:
    objs = OrderedDict(sorted(objs.items(), key=lambda item: len(item[0])))
print(TableFiller(objs))
