import  argparse, time
import numpy as np

from collections import Counter
from itertools import  product
from sklearn.metrics import f1_score

# this function provided by assignment requirement is to load data and format data
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, labels = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_labels = [target_vocab[w.strip()] if to_idx else w.strip() for w in labels.split()]
            inputs.append(words)
            targets.append(ner_labels)
            zip_inps.append(list(zip(words, ner_labels)))
    return zip_inps if as_zip else (inputs, targets)


# this function is to count the current word-current label
def get_cw_cl(corpus):
    corpus_list = []
    for s in corpus:
        corpus_list.extend([d[0] + "_" + d[1] for d in s])
    dict_count = dict(Counter(corpus_list))
    # dict_count = dict((k, dict_count[k]) for k in dict_count if dict_count[k] >= 3)
    return dict_count


# this function is to meet the requirement of the assignment
# to count the x,y in the the current word and current label
def phi_1(x, y):
    counter_dict = dict(Counter([d[0]+ "_" + d[1] for d in zip(x,y)]))
    return counter_dict


# previous-label and current-label function
def phi_2(x,y):
    l = []
    y.insert(0,"<s>")
    y.append("</s>")
    for i in range(1,len(y)):
        l.append(x[i-1]+"_"+y[i-1]+"_"+y[i])
    counter_dict = Counter(l)
    return counter_dict


# training function
def train(train_data, weight, fn, iterate_num, term_index, label_index):
    labels = label_index.keys()
    for i in range(iterate_num):
        train_data = np.random.permutation(train_data)
        start = time.clock()
        for j in range(len(train_data)):
            prediction = predict(weight, train_data[j][0], labels, fn, term_index, label_index)
            update = not prediction[1] == train_data[j][1]
            if update:
                for term, pre_label, true_label in zip(prediction[0], prediction[1], train_data[j][1]):
                    weight[label_index[pre_label]][term_index[term]] -= 1
                    weight[label_index[true_label]][term_index[term]] += 1
        weight /= len(term_index.keys()) # average
        print('Cost of iteration %d: %.2fs ' % (i+1,time.clock() - start))
        start = time.clock()
        f1 = valid_data(get_format_data(test_data), weight, phi_1, word_index, label_index)
        print("f1_score: %f" % (f1))
    return weight


# predict function
def predict(weight, s, t, fn, term_index, label_index):
    permutation_s_t = get_permutation_s_t(s,t)
    scores = np.zeros(len(permutation_s_t))

    for i in range(len(permutation_s_t)):
        fn_dict = fn(permutation_s_t[i][0], permutation_s_t[i][1])
        for key in fn_dict:
            t = key.split("_")[0]
            l = key.split("_")[1]
            if(t in term_index):
                scores[i] += weight[label_index[l]][term_index[t]]*fn_dict[key]

    max_index = np.argmax(scores)

    return permutation_s_t[int(max_index)]

# test the data
def valid_data(data, weight, fn, term_index, label_index):
    labels = label_index.keys()
    correct = []
    predicted = []
    for j in range(len(data)):
        prediction = predict(weight, data[j][0], labels, fn, term_index, label_index)
        correct.extend([_ for _ in data[j][1]])
        predicted.extend([_ for _ in prediction[1]])

    return f1_score(correct, predicted, average='micro', labels=list(label_index.keys()))


# to get all situations of combination of word and labels
def get_permutation_s_t(sentence,labels):
    words_labels = []
    permutation_s_t = []
    for w in sentence:
        words_labels.append([t for t in product([w], labels)])

    iterate = words_labels[0]
    for i in range(1, len(words_labels)):
        iterate = product(iterate, words_labels[i])

    c_s,c_t = [],[]

    def get_s_t(t):  # this function is to make the data to be flatten
        if isinstance(t[0], str):
            c_s.append(t[0])
            c_t.append(t[1])
        else:
            get_s_t(t[0])
            get_s_t(t[1])

    for pro_term in iterate:
        get_s_t(pro_term)
        permutation_s_t.append((tuple(c_s), tuple(c_t)))
        c_s, c_t = [], []

    return permutation_s_t


# to get the unique words dictionary which the value is index of word in matrix
# and unique labels dictionary which the value is index of labels in matrix
def get_word_label_keys(data):
    words = []
    labels= []
    for s in data:
        for term in s:
            words.append(term[0])
            labels.append(term[1])
    words = set(words)
    labels = set(labels)

    # to get unique value
    word_index = dict([_ for _ in zip(words, range(len(words)))])
    label_index = dict([_ for _ in zip(labels, range(len(labels)))])
    return word_index, label_index

# format the data for training
def get_format_data(train_data):
    data = []
    for s in train_data:
        x = []
        y = []
        for term in s:
            x.append(term[0])
            y.append(term[1])
        data.append((tuple(x),tuple(y)))

    return tuple(data)

#------------------- main ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('train_file', type=str, help="Training  file")
parser.add_argument('test_file', type=str, help="Testing file")
args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file

# start = time.clock()
# load data
train_data = load_dataset_sents(train_file)
test_data = load_dataset_sents(test_file)

word_index, label_index = get_word_label_keys(train_data)

np.random.seed(666) # set random seed
# for cw_cl training
ITERATE_NUM = 1

# phi_1
format_data = get_format_data(train_data)
weight = np.random.random((len(label_index.keys()), len(word_index.keys())))*0.001 # to give it a initial value
weight = train(format_data, weight, phi_1, ITERATE_NUM, word_index, label_index)

# phi_1 + phi_2


# print('Cost of Training 1: %.2fs \n'%(time.clock()-start))
# start = time.clock()
# i=0
# for t in train_data:
#     if len(t)>2:
#         print(i)
#     i+=1