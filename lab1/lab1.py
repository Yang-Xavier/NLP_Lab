import array, argparse,time, os, re
from collections import Counter
import numpy as np, pandas as pd

# stoplist =["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours     ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

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
    tensors = list()

    #  counter
    features1,keys = counter(files_data, symbol, sign)
    tensors.append(features1)

    return tensors, keys

def counter(files_data, symbol, sign):
    features = list()
    uniq_keys = list()
    for file_data in files_data:
        dict_data = dict(Counter(re.sub("[^\w']"," ",file_data).split()))
        dict_data[symbol] = sign
        features.append(dict_data)
        uniq_keys.extend(dict_data.keys())

    uniq_keys =  np.unique(uniq_keys) # Get all the feature
    return features,uniq_keys

# def divid_data(train_num, valid_num):

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
uniq_keys = np.unique(np.append(pkeys,nkeys))
i = np.where(uniq_keys == SYMBOL)[0][0]
uniq_keys[[0, i]] = uniq_keys[[i, 0]]

# switch the symbol to first
print('Cost of extracting feature: %.2fs'%(time.clock()-start))
start = time.clock()

data_matrix = build_matrix(np.append(pf[0], nf[0]), uniq_keys)
print('Cost of Building matrix: %.2fs'%(time.clock()-start))
start = time.clock()

data_matrix = np.random.permutation(data_matrix) # random
training_data, valid_data = divide_data(data_matrix, VALID_NUM)
print('Cost of divide training and valid data: %.2fs'%(time.clock()-start))
start = time.clock()










