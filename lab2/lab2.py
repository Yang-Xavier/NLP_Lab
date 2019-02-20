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
            mdata = bi_data.tocsr()
            return mdata[keys_dict[term[0]], keys_dict[term[1]]]
        else:
            return 0
    return look_up


def bigram_LM(uni_keys_dict, uni_data):
    V = len(uni_keys_dict.keys())

    def model(s, smoothing = 0):
        bi_dict = {}
        pro_all = 1
        uni_counter = count_unigram(uni_keys_dict, uni_data)
        bi_counter = count_bigram(uni_keys_dict, bi_data)
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


def process_questions(fdata):
    options=[]
    indices = []
    for line in fdata:
        options.append((line[-2], line[-3]))
        del line[-2]
        del line[-2]

        for i in range(len(line)):
            if line[i] == "____":
                indices.append(i)
                continue
    return options,indices


def answer_questions(uni_keys_dict, uni_data,q_data, options, indices, approach, smoothing = 0):
    answers = []
    if approach == 'unigram':
        uni_counter = count_unigram(uni_keys_dict, uni_data)
        for terms in options:
            answer = terms[0] if uni_counter(terms[0]) > uni_counter(terms[1]) else terms[1]
            answers.append(answer)
    if approach == 'bigram':
        bigram_lm = bigram_LM(uni_keys_dict, uni_data)

        for qi in range(len(q_data)):
            print("Question", qi+1)
            q_data[qi][indices[qi]] = options[qi][0]
            s1,p1 = bigram_lm(q_data[qi], smoothing=smoothing)
            q_data[qi][indices[qi]] = options[qi][1]
            s2,p2 = bigram_lm(q_data[qi], smoothing=smoothing)
            print((q_data[qi][indices[qi] - 1], options[qi][0]), ":", s1[(q_data[qi][indices[qi] - 1], options[qi][0])])
            print((options[qi][0], q_data[qi][indices[qi] + 1]),":", s1[(options[qi][0], q_data[qi][indices[qi] + 1])])
            print("Sentence probability: ", p1 if p1<1 else (p1,"(overflow)"))
            print((q_data[qi][indices[qi] - 1], options[qi][1]), ":", s2[(q_data[qi][indices[qi] - 1], options[qi][1])])
            print((options[qi][1], q_data[qi][indices[qi] + 1]), ":", s2[(options[qi][1], q_data[qi][indices[qi] + 1])])
            print("Sentence probability: ", p2 if p2<1 else "%f (overflow)" % p2)
            answer = options[qi][0] if p1>p2 else options[qi][1]
            answers.append(answer)

    return answers

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

print("Reading the questions")
questions_data = read_file(questions_file)
options, indices = process_questions(questions_data)
print("Answering the questions-----------------------------")
start = time.clock()

print("Using Uni-Gram")
answers = answer_questions(uni_keys_dict, uni_data,questions_data, options, indices, 'unigram')
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using Bi-Gram")
answers = answer_questions(uni_keys_dict, uni_data,questions_data, options, indices, 'bigram')
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using Bi-Gram and add-1 smoothing")
answers = answer_questions(uni_keys_dict, uni_data,questions_data, options, indices, 'bigram', smoothing=1)
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
