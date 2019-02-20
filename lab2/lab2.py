import argparse,time,os,re
from collections import Counter
import multiprocessing as mp
from scipy.sparse import coo_matrix
import numpy as np

MAX_PROCESSORS = os.cpu_count()

def read_file(fname):
    file_data = list()
    with open(fname) as f:
        for line in f:
            line = f.readline()
            new_line =  re.sub("[^\w']"," ", line)
            file_data.append(new_line.split())
    return  file_data

def build_unigram_model(data):
    data_list = []
    for line in data:
        data_list.extend(line)
    data_dict = dict(Counter(data_list))
    keys = np.array(list(data_dict.keys()))
    uni_data = np.array(list(data_dict.values()))
    return keys, uni_data


def MPBigram(data):
    row = []
    col = []

    for line in data:
        line.insert(0, "<s>")
        line.append("</s>")
    i = 0
    for line in data:
        i+=1
        for j in range(len(line)):
            if j + 1 < len(line):
                row.append(np.where(uni_keys == line[j])[0])
                col.append(np.where(uni_keys == line[j+1])[0])
    print(len(row), len(col))
    return (row,col)

def build_bigram_model(data):
    row = []
    col = []
    t = []

    def mp_call(result):
        t.append(result)
        # row.append(result[0])
        # col.append(result[0])


    line_n_each_process = int(len(data)/MAX_PROCESSORS)
    i_b,i_s = 0,0
    with mp.Pool(MAX_PROCESSORS) as pool:
        for i in range(MAX_PROCESSORS-1):
            i_b = int(i*line_n_each_process)
            i_s = int((i+1)*line_n_each_process)

            pool.apply_async(MPBigram, (data[i_b:i_s],), callback=mp_call)
        pool.apply_async(MPBigram, (data[i_s:],),  callback=mp_call)

        pool.close()
        pool.join()

    # for line in data:
    #     line.insert(0, "<s>")
    #     line.append("</s>")
    # i = 0
    # for line in data:
    #     i += 1
    #     for j in range(len(line)):
    #         if j + 1 < len(line):
    #             row.append(np.where(uni_keys == line[j])[0])
    #             col.append(np.where(uni_keys == line[j+1])[0])
    #
    #     print(i)





# def count_unigram(term):
#
# def count_bigram(term):




parser = argparse.ArgumentParser()
parser.add_argument('t_file', type=str, help="Training model file")
parser.add_argument('q_file', type=str, help="Question file")
args = parser.parse_args()

training_file = args.t_file
question_file = args.q_file

start = time.clock()
training_data = read_file(training_file)

print('Cost of Reading files: %.2fs'%(time.clock()-start))
start = time.clock()

uni_keys, uni_data=build_unigram_model(training_data)
print('Cost of Building Uni-Gram model: %.2fs'%(time.clock()-start))
start = time.clock()

build_bigram_model(training_data)
print('Cost of Building Bi-Gram model: %.2fs'%(time.clock()-start))
start = time.clock()