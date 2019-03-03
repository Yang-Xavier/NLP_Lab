import argparse,time,re
from collections import Counter
from scipy.sparse import *
import numpy as np
import sys

def read_file(fname):
    file_data = list()
    with open(fname) as f:
        line = f.readline()
        while line:     # read file and parse it into list array
            new_line =  "<s> " + re.sub("\s[^\w\s]+[$|\s]|[^\w\s]{2,}"," ", line.lower()) + " </s>"
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
    keys_dict = {}  # give each key feature a number that will be used as index in matrix
    for i in range(len(keys)):
        keys_dict[keys[i]] = i+1
    return keys_dict, uni_data


def build_bigram_matrix(data, uni_keys_dict, data_type = np.float64): # to build the matrix
    row = []
    col = []
    # for sparse matrix coo_matrix (data,(row,col))
    for line in data:
        for j in range(len(line)):
            if j + 1 < len(line):
                r = uni_keys_dict[line[j]]
                c = uni_keys_dict[line[j+1]]
                row.append(r)
                col.append(c)

    # build sparse matrix
    row = np.asarray(row)
    col = np.asarray(col)
    d = np.ones((1, len(col))).flatten()
    bi_matrix = coo_matrix((d,(row,col)), dtype=data_type)

    return bi_matrix


def count_unigram(keys_dict, uni_data): # procedure oriented store the variable in local and private
    def  look_up(term):
        if term in keys_dict:
            return uni_data[keys_dict[term]]
        else:
            return 0

    return look_up

def count_bigram(keys_dict, bi_matrix): # same
    mdata = bi_matrix.todok()

    def look_up(term):  #
        if term[0] in keys_dict and term[1] in keys_dict:
            return mdata[keys_dict[term[0]], keys_dict[term[1]]]
        else:
            return 0
    return look_up


def bigram_LM(uni_keys_dict, uni_data, bi_matrix ): # same
    V = len(uni_keys_dict.keys())
    uni_counter = count_unigram(uni_keys_dict, uni_data)
    bi_counter = count_bigram(uni_keys_dict, bi_matrix)

    def model(s, smoothing = 0):    # bigram language model, s is the sentence array list
        bi_dict = {}
        pro_all = 1

        for i in range(len(s)):
            if i+1 < len(s):
                b = (uni_counter(s[i])+smoothing*V)
                if b == 0:
                    pro = 0
                else:
                    pro = (bi_counter((s[i], s[i+1]))+smoothing)/b

                bi_dict[(s[i], s[i + 1])] = pro
                pro_all *= pro

        return bi_dict,pro_all

    return model

# pre-process the question data, split the option and blank location
def process_questions(fdata):
    options=[]
    indices = []
    for line in fdata:
        options.append(line[-2].split("/"))
        del line[-2]
        del line[-2]

        for i in range(len(line)):
            if line[i] == "____":
                indices.append(i)
                continue
    return options,indices


def answer_questions(uni_keys_dict, uni_data, bi_matrix, q_data, options, indices, approach, smoothing = 0 ):
    answers = []
    if approach == 'unigram':
        uni_counter = count_unigram(uni_keys_dict, uni_data)
        for terms in options:
            count_a = uni_counter(terms[0])
            count_b = uni_counter(terms[1])
            if count_a == 0. and count_b == 0.:
                answer = "_NULL_"
            elif count_a == count_b :
                answer = "_HALF_"
            else:
                answer = terms[0] if count_a > count_b else terms[1]

            answers.append(answer)
    if approach == 'bigram':
        bigram_lm = bigram_LM(uni_keys_dict, uni_data, bi_matrix)

        for qi in range(len(q_data)): # question data
            question = q_data[qi]
            question[indices[qi]] = options[qi][0]    # Using the answer to  replace the blank in the question
            s1,p1 = bigram_lm(q_data[qi], smoothing=smoothing)
            question[indices[qi]] = options[qi][1]
            s2,p2 = bigram_lm(q_data[qi], smoothing=smoothing)

            if p1 == 0. and p2 == 0.:
                answer = "_NULL_"
            elif p1 == p2 :
                answer = "_HALF_"
            else:
                answer = options[qi][0] if p1 > p2 else options[qi][1]

            answers.append(answer)

    return answers


def calculate_accuracy(answer) :
    stand_answer = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']
    accuracy = 0
    for i in range(len(stand_answer)):
        if answer[i] == stand_answer[i]:
            accuracy += 1
        elif answer[i] == '_HALF_':
            accuracy += 0.5

    return accuracy / len(stand_answer)

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
print('Cost of Building unigram model: %.2fs'%(time.clock()-start))
start = time.clock()

bi_matrix = build_bigram_matrix(training_data, uni_keys_dict)
print('Cost of Building bigram model: %.2fs'%(time.clock()-start))

questions_data = read_file(questions_file)
options, indices = process_questions(questions_data)

print('coo_matrix', sys.getsizeof(bi_matrix.data))
print('dok_matrix', sys.getsizeof(bi_matrix.todok()))
print('2D-array', sys.getsizeof(bi_matrix.toarray()))

print("Answering the questions-----------------------------")
start = time.clock()

print("Using unigram")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix, questions_data, options, indices, 'unigram')
print("Answers: ", answers)
print('Accuracy: %.2f' % calculate_accuracy(answers))
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using bigram")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix,questions_data, options, indices, 'bigram')
print("Answers: ", answers)
print('Accuracy: %.2f' % calculate_accuracy(answers))
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using bigram and add-1 smoothing")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix, questions_data, options, indices, 'bigram', smoothing=1)
print("Answers: ", answers)
print('Accuracy: %.2f' % calculate_accuracy(answers))
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
