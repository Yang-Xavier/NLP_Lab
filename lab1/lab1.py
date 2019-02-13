import argparse,time, os, re
from collections import Counter
import numpy as np

# read the files into list and remove the space an \n
def read_files(folder):
    files = os.listdir(folder)
    file_data = list()
    for file in files:
        with open(folder+"/"+file) as f:
            file_data.append(f.read())

    return file_data

# Go through all the file list by using the extract_functions
def extract_feature(files_data, symbol,sign):
    features_tensors = list()
    keys_tensors = list()
    #  counter
    features1,keys1 = counter_unigram(files_data, symbol, sign)
    features_tensors.append(features1)
    keys_tensors.append(keys1)
    return features_tensors, keys_tensors

def counter_unigram(files_data, symbol, sign):
    features = list()
    uniq_keys = list()
    for file_data in files_data:
        dict_data = dict(Counter(re.sub("[^\w']"," ",file_data).split()))
        dict_data[symbol] = sign
        features.append(dict_data)
        uniq_keys.extend(dict_data.keys())

    uniq_keys =  np.unique(uniq_keys) # Get all the feature
    return features,uniq_keys

def counter_bigram(files_data, symbol, sign):

# def counter_trigram(files_data, symbol, sign):


def build_matrix(data, keys):
    matrix = np.zeros((len(data), len(keys)))
    c_index = 0
    for term in keys:
        for index in range(len(data)):
            if term in data[index]:
                matrix[index][c_index] = data[index][term]
        c_index+=1
    return matrix

def divide_data(data, valid_num, random=False):
    pos_n = int(valid_num / 2)
    neg_n = int(valid_num - pos_n)

    valid_data_index = np.append(np.where(data[:,0]>0)[0][:pos_n], np.where(data[:,0]<0)[0][:neg_n])
    valid_data = data[valid_data_index]
    training_data = np.delete(data, valid_data_index, axis=0)

    return training_data,valid_data

def validation(weight, valid_data):
    positive_data = valid_data[np.where(valid_data[:,0]>0)[0]]
    negative_data = valid_data[np.where(valid_data[:,0]<0)[0]]
    pre_positive = np.sign(np.dot(np.delete(positive_data, [0], axis=1), weight))
    pre_negative = np.sign(np.dot(np.delete(negative_data, [0], axis=1), weight))

    true_positive = (pre_positive>0).sum()
    false_positive = positive_data.shape[0] - true_positive
    true_negative = (pre_negative < 0).sum()
    false_negative = negative_data.shape[0] - true_negative

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive + false_negative)

    return true_positive,true_negative,precision,recall

# main
start = time.clock()
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help="folder", default="review_polarity")
args = parser.parse_args()

folder_name = args.folder if 'folder' in args else 'review_polarity'
folder_name += "/txt_sentoken"
pos_folder = folder_name+"/pos"
neg_folder = folder_name+"/neg"

POS = 1
NEG = -1
SYMBOL = "_sign_"
VALID_NUM = 200

pos_files = read_files(pos_folder)
neg_files = read_files(neg_folder)

print('Cost of Reading files: %.2fs'%(time.clock()-start))
start = time.clock()

pf, pkeys = extract_feature(pos_files, SYMBOL, POS)
nf, nkeys = extract_feature(neg_files, SYMBOL, NEG)
uniq_keys = np.unique(np.append(pkeys[0],nkeys[0]))
i = np.where(uniq_keys == SYMBOL)[0][0]
uniq_keys[[0, i]] = uniq_keys[[i, 0]]

# switch the symbol to first
print('Cost of Extracting feature: %.2fs'%(time.clock()-start))
start = time.clock()

data_matrix = build_matrix(np.append(pf[0], nf[0]), uniq_keys)
print('Cost of Building matrix: %.2fs'%(time.clock()-start))
start = time.clock()

data_matrix = np.random.permutation(data_matrix) # random
training_data, valid_data = divide_data(data_matrix, VALID_NUM)
print('Cost of Dividing training and valid data: %.2fs'%(time.clock()-start))
start = time.clock()

#  start iteration
signs = training_data[:,0].reshape((training_data.shape[0],1))
training_data = np.delete(training_data,[0],axis=1)
weight = np.zeros((training_data.shape[1], 1))
MAX_ITERATION = 1000

# print('Start Training-----------------------------------------------------------------------------')
# error = training_data.shape[0]
# i = 0
# while(error>=200):
#     pre = np.sign(np.dot(training_data, weight))
#     acc_index = np.where(pre != signs)[0]
#     acc_vec = np.zeros((training_data.shape[0],1))
#     acc_vec[acc_index] = signs[acc_index]
#     acc = np.dot(training_data.T, acc_vec) / training_data.shape[0]
#     weight += acc
#     error = (np.abs(signs - pre)).sum()
#     if (i+1)%50==0:
#         print("Iteraton %d : Error  %d  " % (i+1, error))
#     i+=1
# print('Cost of Training: %.2f s'%(time.clock()-start))
#
# true_positive,true_negative,precision,recall = validation(weight, valid_data)
# print("TruePositive: %d \n TrueNegative: %d \n Precision: %.2f \n Recall: %.2f \n" % (true_positive, true_negative, precision, recall))
