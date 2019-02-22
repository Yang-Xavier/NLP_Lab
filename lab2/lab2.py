import argparse,time,re
from collections import Counter
from scipy.sparse import coo_matrix
import numpy as np

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


def build_bigram_matrix(data, uni_keys_dict, data_type = np.float32): # to build the matrix
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
    mdata = bi_matrix.tocsr()
    def look_up(term):  #
        if term[0] in keys_dict and term[1] in keys_dict:
            return mdata[keys_dict[term[0]], keys_dict[term[1]]]
        else:
            return 0
    return look_up


def bigram_LM(uni_keys_dict, uni_data, bi_matrix ): # same
    V = len(uni_keys_dict.keys())

    def model(s, smoothing = 0):    # bigram language model, s is the sentence array list
        bi_dict = {}
        pro_all = 1
        uni_counter = count_unigram(uni_keys_dict, uni_data)
        bi_counter = count_bigram(uni_keys_dict, bi_matrix)

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


def process_questions(fdata):   # pre-process the question data, split the option and blank location
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


def answer_questions(uni_keys_dict, uni_data, bi_matrix, q_data, options, indices, approach, log, smoothing = 0 ):
    answers = []
    if approach == 'unigram':
        uni_counter = count_unigram(uni_keys_dict, uni_data)
        for terms in options:
            answer = terms[0] if uni_counter(terms[0]) > uni_counter(terms[1]) else terms[1]
            answers.append(answer)
    if approach == 'bigram':
        bigram_lm = bigram_LM(uni_keys_dict, uni_data, bi_matrix)

        for qi in range(len(q_data)): # question data

            q_data[qi][indices[qi]] = options[qi][0]
            s1,p1 = bigram_lm(q_data[qi], smoothing=smoothing)
            q_data[qi][indices[qi]] = options[qi][1]
            s2,p2 = bigram_lm(q_data[qi], smoothing=smoothing)

            log["push"]("Question %d" % (qi + 1))
            log["push"]("%s : %f" % ((q_data[qi][indices[qi] - 1], options[qi][0]), s1[(q_data[qi][indices[qi] - 1], options[qi][0])]))
            log["push"]("%s : %f" % ((options[qi][0], q_data[qi][indices[qi] + 1]), s1[(options[qi][0], q_data[qi][indices[qi] + 1])]))
            log["push"]("Sentence probability : %s" % (p1 if p1<1 else "%f (overflow)" % p1))
            log["push"]("%s : %f" % ((q_data[qi][indices[qi] - 1], options[qi][1]), s2[(q_data[qi][indices[qi] - 1], options[qi][1])]))
            log["push"]("%s : %f" % ((options[qi][1], q_data[qi][indices[qi] + 1]), s2[(options[qi][1], q_data[qi][indices[qi] + 1])]))
            log["push"]("Sentence probability : %s" % (p2 if p2<1 else "%f (overflow)" % p2))

            answer = options[qi][0] if p1>p2 else options[qi][1]
            answers.append(answer)

    return answers

def log():
    l = list()
    def push(string):
        l.append(string)

    def print_all():
        for s in l:
            print(s)
        l.clear()

    return {"push": push, "print_all": print_all}

# main
parser = argparse.ArgumentParser()
parser.add_argument('t_file', type=str, help="Training model file")
parser.add_argument('q_file', type=str, help="Question file")
args = parser.parse_args()
log_list = log()

training_file = args.t_file
questions_file = args.q_file

start = time.clock()
training_data = read_file(training_file)
print('Cost of Reading files: %.2fs'%(time.clock()-start))
start = time.clock()

uni_keys_dict, uni_data=build_unigram_model(training_data)

print('Cost of Building Uni-Gram model: %.2fs'%(time.clock()-start))
start = time.clock()

bi_matrix = build_bigram_matrix(training_data, uni_keys_dict)
print('Cost of Building Bi-Gram model: %.2fs'%(time.clock()-start))

print("Reading the questions")
questions_data = read_file(questions_file)
options, indices = process_questions(questions_data)
print("Answering the questions-----------------------------")
start = time.clock()

print("Using Uni-Gram")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix, questions_data, options, indices, 'unigram', log = log_list)
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using Bi-Gram")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix,questions_data, options, indices, 'bigram', log = log_list)
log_list["print_all"]()
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
start = time.clock()

print("Using Bi-Gram and add-1 smoothing")
answers = answer_questions(uni_keys_dict, uni_data, bi_matrix, questions_data, options, indices, 'bigram', smoothing=1, log = log_list)
log_list["print_all"]()
print("Answers: ", answers)
print('Cost of Answering the questions: %.2fs \n'%(time.clock()-start))
