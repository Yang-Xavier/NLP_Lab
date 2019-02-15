import argparse,time, os, re
from collections import Counter
import numpy as np

# read the files into list and remove the space an \n
def read_files(folder):
    files = os.listdir(folder)
    file_data = list()
    for file in files:
        with open(folder+"/"+file) as f:
            file_data.append(re.sub("[^\w']"," ",f.read()).split())  # read the file to list
    return file_data

# Go through all the file list by using the extract_functions
def extract_feature(files_data, symbol,sign, type = 'unigram'):
    features, keys = list(),list()
    #  counter
    if type == 'unigram':
        features,keys = counter_unigram(files_data, symbol, sign)
    if type == 'bigram':
        features,keys = counter_bigram(files_data, symbol, sign)
    if type == 'trigram':
        features,keys = counter_trigram(files_data, symbol, sign)

    return features, keys

def counter_unigram(files_data, symbol, sign):
    features = list()
    uniq_keys = list()
    for file_data in files_data:
        dict_data = dict(Counter(file_data))
        dict_data[symbol] = sign
        features.append(dict_data)
        uniq_keys.extend(dict_data.keys())

    uniq_keys =  np.unique(uniq_keys) # Get all the feature
    return features,uniq_keys

def counter_bigram(files_data, symbol, sign):
    features = list()
    uniq_keys = list()
    for file_data in files_data:
        single = list()
        for i in range(len(file_data)):
            if i+1 < len(file_data):
                single.append(file_data[i] + " " + file_data[i+1])
        dict_data = dict(Counter(single))
        dict_data[symbol] = sign
        features.append(dict_data)
        uniq_keys.extend(dict_data.keys())

    uniq_keys =  np.unique(uniq_keys) # Get all the feature
    return features,uniq_keys

def counter_trigram(files_data, symbol, sign):
    features = list()
    uniq_keys = list()
    for file_data in files_data:
        single = list()
        for i in range(len(file_data)):
            if i+1 < len(file_data) and i+2 < len(file_data):
                single.append(file_data[i] + " " + file_data[i+1] + " " + file_data[i+2])
        dict_data = dict(Counter(single))
        dict_data[symbol] = sign
        features.append(dict_data)
        uniq_keys.extend(dict_data.keys())

    uniq_keys =  np.unique(uniq_keys) # Get all the feature
    return features,uniq_keys

# According to the unique keys to build up the matrix
def build_matrix(data, keys):
    matrix = np.zeros((len(data), len(keys)))
    c_index = 0
    for term in keys:
        for index in range(len(data)):
            if term in data[index]:
                matrix[index][c_index] = data[index][term]
        c_index+=1
    return matrix

# Divide the data into valid data set and training data set
def divide_data(data, valid_num, random=False):
    pos_n = int(valid_num / 2)
    neg_n = int(valid_num - pos_n)

    valid_data_index = np.append(np.where(data[:,0]>0)[0][:pos_n], np.where(data[:,0]<0)[0][:neg_n])
    valid_data = data[valid_data_index]
    training_data = np.delete(data, valid_data_index, axis=0)

    return training_data,valid_data

# Do test for the training weight
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

def print_top_ten(weight, keys):
    sort_index = np.argsort(weight, axis=0)
    positive_top_ten = sort_index[-10:]
    negative_top_ten = sort_index[:10]

    print("Top ten weight for positive and negative: \nPositive_term: Weight  \t Negative_term: Weight")
    for i in range(10):
        print("%s: %.4f \t %s: %.4f" % (keys[positive_top_ten[-i]][0], weight[positive_top_ten[-i]], keys[negative_top_ten[i]][0], weight[negative_top_ten[i]]))

def run(approach):
    start = time.clock()
    print('\n----------------------------------%s-----------------------------------------'%(approach))
    pf, pkeys = extract_feature(pos_files, SYMBOL, POS, approach)
    nf, nkeys = extract_feature(neg_files, SYMBOL, NEG, approach)
    uniq_keys = np.unique(np.append(pkeys, nkeys))
    i = np.where(uniq_keys == SYMBOL)[0][0]
    uniq_keys[[0, i]] = uniq_keys[[i, 0]]

    # switch the symbol to first
    print('Cost of Extracting feature: %.2fs' % (time.clock() - start))
    start = time.clock()

    data_matrix = build_matrix(np.append(pf, nf), uniq_keys)
    print('Cost of Building matrix: %.2fs' % (time.clock() - start))
    print('The length of the features:  ', uniq_keys.shape[0])
    start = time.clock()

    training_data, valid_data = divide_data(data_matrix, VALID_NUM)
    print('Cost of Dividing training and valid data: %.2fs' % (time.clock() - start))
    start = time.clock()

    #  start iteration

    weight = np.zeros((training_data.shape[1] - 1, 1))
    signs = training_data[:, 0].reshape((training_data.shape[0], 1))
    MAX_ITERATION = 10
    error = training_data.shape[0]

    print('Start Training-----------------------------------------------------------------------------')

    training_data_ = training_data  # backup training data

    for i in range(MAX_ITERATION):
        np.random.seed(i)
        training_data = np.random.permutation(training_data_)
        signs = training_data[:, 0].reshape((training_data.shape[0], 1))
        training_data = training_data[:, 1:]

        for j in range(training_data.shape[0]):
            pre = 1 if np.sign(np.dot(training_data[j], weight))[0] >= 0 else -1
            if pre != signs[j]:
                weight += (signs[j] * training_data[j]).reshape(weight.shape[0], 1)
        pre_all = np.sign(np.dot(training_data, weight))
        error = (np.abs(signs - pre_all)).sum()
        print("Iteraton %d : Error  %d  " % (i + 1, error))

    weight /= training_data.shape[0]
    print('End Training-------------------------------------------------------------------------------')
    print('Cost of Training: %.2fs' % (time.clock() - start))

    true_positive, true_negative, precision, recall = validation(weight, valid_data)
    print("TruePositive: %d \nTrueNegative: %d \nPrecision: %.2f \nRecall: %.2f \n" % (
    true_positive, true_negative, precision, recall))

    print_top_ten(weight, uniq_keys)

# main
start = time.clock()
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help="folder", default="review_polarity")
parser.add_argument('-t', type=str, help="term", default="all", required=False)
args = parser.parse_args()

folder_name = args.folder if 'folder' in args else 'review_polarity'
folder_name += "/txt_sentoken"
pos_folder = folder_name+"/pos"
neg_folder = folder_name+"/neg"

POS = 1
NEG = -1
SYMBOL = "_sign_"
VALID_NUM = 400
FEATURE_TYPE = 'all' if  not args.t else args.t

pos_files = read_files(pos_folder)
neg_files = read_files(neg_folder)

print('Cost of Reading files: %.2fs'%(time.clock()-start))
start = time.clock()

if FEATURE_TYPE is not 'all':
    run(FEATURE_TYPE)
else:
    run('unigram')
    run('bigram')
    run('trigram')
