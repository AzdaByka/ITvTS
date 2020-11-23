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


def import_documents(documents_name, directory):
    dataset = []
    for document in documents_name:
        dataset.append(get_document_data(document,directory))
    return dataset


def get_document_data(file,directory):
    with open(directory+"/" + file + ".txt", "r", encoding="utf-8") as file:
        return str(file.readlines());


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
    objs = defaultdict(list)
    lengths_of_documents = []
    id_document = 0
    for inp in dataset:
        inp = inp.lower()
        inp = special_symbol_replace(inp)
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
        objs = remove_empties(objs)
        id_document += 1
    return objs, lengths_of_documents

def get_lemma(dataset):
    objs = defaultdict(list)
    inp = dataset.lower()
    inp = special_symbol_replace(inp)
    inp = punctuation_replace(inp)
    inp = re.sub('[-«»—]', '', inp)
    doc = tokenizer(inp)
    for token in list(doc):
        token = token.lemma_
        if token not in objs.keys():
                objs[token].append(0)
    objs = remove_empties(objs)
    return objs

# endregion

# region Bayes

def get_count_word_in_corpus(lengths_of_documents):
    length = 0
    for item in lengths_of_documents:
        length += item
    return length


def get_count_word_in_classes(lengths_of_documents):
    count_word_in_classes = [0, 0, 0]
    for i in range(0, 3):
        count_word_in_classes[0] += lengths_of_documents[i]
    for i in range(3, 6):
        count_word_in_classes[1] += lengths_of_documents[i]
    for i in range(6, 9):
        count_word_in_classes[2] += lengths_of_documents[i]
    return count_word_in_classes


def calculate_count_wc(vectors):
    vectors_count_ws = defaultdict(list)
    for token in vectors:
        for i in range(0, 3):
            vectors_count_ws[token].append(0)
        for i in range(0, 3):
            vectors_count_ws[token][0] += vectors[token][i]
        for i in range(3, 6):
            vectors_count_ws[token][1] += vectors[token][i]
        for i in range(6, 9):
            vectors_count_ws[token][2] += vectors[token][i]
    return vectors_count_ws;


def calculate_pwc_for_each_class(vectors_count_ws, count_word_in_corpus, count_word_in_classes):
    vectors_pws = defaultdict(list)
    for token in vectors_count_ws:
        for i in range(0, 3):
            vectors_pws[token].append(0)
        for i in range(0, 3):
            vectors_pws[token][i] = (vectors_count_ws[token][i] + 1) / (count_word_in_corpus + count_word_in_classes[i])
    return vectors_pws


def calculate_pcd(vectors_request, vectors_pws_database):
    pcd=[0,0,0]
    pc=math.log(1/3)
    for token in vectors_request:
        for i in range(0,3):
            if token[0] in vectors_pws_database.keys():
                pcd[i] += math.log(vectors_pws_database[token[0]][i])
    for i in range(0, 3):
        pcd[i] += pc
    return pcd;

# endregion

def table_filler(objs):
    x = PrettyTable()
    x.field_names = ['pcd1', 'pcd2', 'pcd3']
    x.add_row([str(objs[0]), str(objs[1]), str(objs[2])])
    return x


def main():
    documents_name = ['C1_1', 'C1_2', 'C1_3', 'C2_1', 'C2_2', 'C2_3', 'C3_1', 'C3_2', 'C3_3'];
    dataset = import_documents(documents_name,"train")
    vectors_frequencies, lengths_of_documents = Get_vector_frequency(dataset);
    count_word_in_corpus = get_count_word_in_corpus(lengths_of_documents);
    count_word_in_classes = get_count_word_in_classes(lengths_of_documents);
    vectors_count_ws = calculate_count_wc(vectors_frequencies);
    vectors_pws = calculate_pwc_for_each_class(vectors_count_ws, count_word_in_corpus, count_word_in_classes);
    #print(table_filler(vectors_pws))
    # print("Найдено уникальных слов "+vectors_count_ws.__len__())
    documents_name = ['unknown1', 'unknown2', 'unknown3'];
    dataset = get_document_data('unknown3','test')
    vectors_frequencies = get_lemma(dataset);
    pcd = calculate_pcd(vectors_frequencies, vectors_pws)
    print(table_filler(pcd))


main()
