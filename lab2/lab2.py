import argparse,time,re
from collections import Counter
from scipy.sparse import coo_matrix
import numpy as np

def read_file(fname):
    file_data = list()
    with open(fname) as f:
        line = f.readline()
        while line:
            new_line =  "<s> " + re.sub("[^\w']"," ", line) + " </s>"
            file_data.append(new_line.split())
            line = f.readline()
    return  file_data

def build_unigram_model(data):
    data_list = []
    for line in data:
        data_list.extend(line)
    data_dict = dict(Counter(data_list))
    keys = np.array(list(data_dict.keys()))
    uni_data = np.array(list(data_dict.values()))
    keys_dict = {}
    for i in range(len(keys)):
        keys_dict[keys[i]] = i+1
    return keys_dict, uni_data


def build_bigram_model(data, uni_keys_dict, data_type = np.float32):
    row = []
    col = []

    for line in data:
        for j in range(len(line)):
            if j + 1 < len(line):
                r = uni_keys_dict[line[j]]
                c = uni_keys_dict[line[j+1]]
                row.append(r)
                col.append(c)

    # build spare matrix
    row = np.asarray(row)
    col = np.asarray(col)
    d = np.ones((1, len(col))).flatten()
    bi_matrix = coo_matrix((d,(row,col)), dtype=data_type)

    return bi_matrix


def count_unigram(keys_dict, uni_data):
    def  look_up(term):
        if term in keys_dict:
            return uni_data[keys_dict[term]]
        else:
            return 0

    return look_up

def count_bigram(keys_dict, bi_data):
    def look_up(term):
        if term[0] in keys_dict and term[1] in keys_dict:
            return bi_data[keys_dict[term[0]], keys_dict[term[1]]]
        else:
            return 0

    return look_up

def process_questions(fdata):
    options=[]
    indices = []
    for line in fdata:
        options.append((line[-2], line[-3]))
        for i in range(len(line)):
            if line[i] == "____":
                indices.append(i)
                continue
    return options,indices


def answer_questions(q_data, options, indices, approach):
    return

# main
parser = argparse.ArgumentParser()
parser.add_argument('t_file', type=str, help="Training model file")
parser.add_argument('q_file', type=str, help="Question file")
args = parser.parse_args()

training_file = args.t_file
questions_file = args.q_file

start = time.clock()
training_data = read_file(training_file)
print('Cost of Reading files: %.2fs'%(time.clock()-start))
start = time.clock()

uni_keys_dict, uni_data=build_unigram_model(training_data)
print('Cost of Building Uni-Gram model: %.2fs'%(time.clock()-start))
start = time.clock()

bi_data = build_bigram_model(training_data, uni_keys_dict)
print('Cost of Building Bi-Gram model: %.2fs'%(time.clock()-start))
start = time.clock()

print("Building the counter for Uni-Gram and Bi-Gram")
uni_counter = count_unigram(uni_keys_dict, uni_data)
bi_counter = count_bigram(uni_keys_dict, bi_data)

print("Reading the questions")
questions_data = read_file(questions_file)
options, indices = process_questions(questions_data)
print("Answering the questions-----------------------------")
print("Using Uni-Gram")


