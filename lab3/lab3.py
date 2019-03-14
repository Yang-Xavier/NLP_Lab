import  argparse, time
import numpy as np

from collections import Counter
from itertools import  product
from sklearn.metrics import f1_score
from scipy.sparse import coo_matrix

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


# training function
def train(train_data,  weight, keys, labels,iterate_num, phi_n , valid_func = None):
    weight_sum  = weight
    for i in range(iterate_num):
        start = time.clock()
        np.random.seed(i)
        train_data = np.random.permutation(train_data)
        for j in range(len(train_data)):
            prediction = predict(weight, train_data[j][0], labels, keys, phi_n)
            update = not prediction[1] == train_data[j][1]

            if update:
                weight = update_weight(weight, train_data[j], prediction, keys, phi_n)

        weight_sum += weight  # sum

        print('Cost of %d th iteration: %.2fs ' % ( i+1 ,(time.clock() - start)))

        if(valid_func):
            f1_score = valid_func(weight, labels, keys, phi_n)
            print("f1_score: %f \n" % f1_score)

        start = time.clock()

    return weight_sum / iterate_num


def update_weight(weight, corrected, predicted, keys, phi_n):
    for phi in phi_n:
        co_dict = phi(corrected[0],corrected[1])
        pr_dict = phi(predicted[0],predicted[1])
        for k in co_dict:
            weight[keys[k]] += 1
        for k in pr_dict:
            weight[keys[k]] -= 1
    return weight


# predict function
def predict(weight, sentence,  labels, keys, phi_n):
    permutation_data = get_permutation(sentence, labels)
    matrix = building_matrix(permutation_data, phi_n, keys)
    r = matrix.dot(weight)
    max_index = np.argmax(r)

    return permutation_data[int(max_index)]


# to build the sparse matrix
def building_matrix(permutation_data, phi_n, term_keys):
    row = []
    col = []
    data = []
    for i in range(len(permutation_data)):
        for phi in phi_n:
            fn_dict = phi(permutation_data[i][0], permutation_data[i][1])
            for k in fn_dict:
                if k in term_keys:
                    row.append(i)
                    col.append(term_keys[k])
                    data.append(fn_dict[k])

    return coo_matrix((data, (row, col)), shape=(len(permutation_data), len(term_keys.keys())))


# to get all situations of combination of terms and label
def get_permutation(terms, labels):
    term_labels = []
    permutation_s_t = []
    c_w, c_l = [], []  # current word, current label

    for w in terms:
        term_labels.append([l for l in product([w], labels)])

    iterate = term_labels[0]
    for i in range(1, len(term_labels)):
        iterate = product(iterate, term_labels[i])


    def flatten(terms):  # this function is to make the data to be flatten e.g. ((("a"),"b"),"c") ==> ("a","b","c")
        if isinstance(terms[0], str):
            c_w.append(terms[0])
            c_l.append(terms[1])
        else:
            flatten(terms[0])
            flatten(terms[1])

    for pro_term in iterate:
        flatten(pro_term)
        permutation_s_t.append((tuple(c_w), tuple(c_l)))
        c_w, c_l = [], []

    return permutation_s_t


# test the data
def valid_data(test_data):

    correct = []
    predicted = []

    def valid_ (weight, labels, keys, phi_n):
        for j in range(len(test_data)):
            prediction = predict(weight, test_data[j][0], labels, keys, phi_n)
            correct.extend([_ for _ in test_data[j][1]])
            predicted.extend([_ for _ in prediction[1]])
        return f1_score(correct, predicted, average='micro', labels=list(labels))

    return  valid_


# this function is to meet the requirement of the assignment
# to count the x,y in the the current word and current label
def phi_1(x, y):
    return dict(Counter([d[0]+ "_" + d[1] for d in zip(x,y)]))


# previous-label and current-label
def phi_2(x,y):
    l = []
    y = list(y)
    y.insert(0,"NULL")

    for i in range(1,len(y)):
        l.append(y[i]+"_"+y[i-1])

    return dict(Counter(l))


# suffix-3 and current label
def phi_3(x,y):
    l = []

    for  i in range(len(x)) :
        if len(x[i]) > 3:
            l.append(x[i][-3:] + "_" + y[i])
    return dict(Counter(l))


# to get the unique words dictionary which the value is index of word in matrix
# and unique labels dictionary which the value is index of labels in matrix
def get_keys_for_phi1(data):
    words = []
    labels= []
    keys = []
    for s in data:
        for term in s:
            words.append(term[0])
            labels.append(term[1])
    words = np.unique(words)
    labels = np.unique(labels)

    for w in words:
        keys.extend(w+"_"+l for l in labels)
    return keys


# similar to the function above, the term is the current word + previous label
# it could be based on the function get_keys_for_phi1
# label_index will not change
def get_keys_for_phi2(data):
    labels = []
    keys = []
    for s in data:
        for term in s:
            labels.append(term[1])
    labels = np.unique(labels)
    for l in labels:
        keys.extend(l + "_" + _ for _ in labels)
        keys.append(l+"_"+"NULL")


    return keys


# take the suffix-3 as the feature
def get_keys_for_phi3(data):
    words = []
    labels = []
    keys = []
    for s in data:
        for term in s:
            if len(term[0]) > 3:
                words.append(term[0][-3:])   # only consider the sentence which has length more than 3
            labels.append(term[1])
    words = np.unique(words)
    labels = np.unique(labels)

    for w in words:
        keys.extend(w + "_" + l for l in labels)

    return keys

# take current word and previous word as feature
# def get_keys_for_phi4(data):
#     words = []
#     keys = []
#     for s in data:
#         for term in s:
#             words.append(term[0])
#
#     words = np.unique(words)
#     for w in words:
#         keys.extend(w + "_" + w_ for w_ in words)
#         keys.append(w + "_<s>" )
#
#     return keys


def combine_keys(ks):
    keys = []
    for k in ks:
        keys.extend(k)

    key_dict ={}# dict([(k,v) for (k,v) in zip(keys, range(len(keys)))])
    for (k, v) in zip(keys, range(len(keys))):
        key_dict[k] = v

    return  key_dict

def get_labels(data):
    labels = []
    for s in data:
        for term in s:
            labels.append(term[1])

    labels = np.unique(labels)
    return labels

# format the data for training, ([sentence],[labels])
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

# set random seed

ITERATE_NUM = 10

labels = get_labels(train_data)

# phi_1
print("Only phi_1 ------------------- ")
phi_1_keys = get_keys_for_phi1(train_data)
keys = combine_keys([phi_1_keys])
weight = np.zeros((len(keys.keys()), 1)) # to give it a initial value
weight = train(get_format_data(train_data), weight, keys, labels,ITERATE_NUM, [phi_1], valid_func = valid_data(get_format_data(test_data)))


# phi_1 + phi_2
print("Combine phi_1 and phi_2------------------- ")
phi_1_keys = get_keys_for_phi1(train_data)
phi_2_keys = get_keys_for_phi2(train_data)
keys = combine_keys([phi_1_keys, phi_2_keys])
weight = np.zeros((len(keys.keys()), 1)) # to give it a initial value
weight = train(get_format_data(train_data), weight, keys, labels,ITERATE_NUM, [phi_1, phi_2], valid_func = valid_data(get_format_data(test_data)))


# phi_1 + phi_2 + phi_3 + phi_4
# term_index, label_index = get_keys_for_phi2(train_data)
# weight = np.zeros((len(label_index.keys()), len(term_index.keys()))) # to give it a initial value
# weight = train(get_format_data_phi2(train_data), weight, ITERATE_NUM, term_index, label_index)
# f1 = valid_data(get_format_data_phi2(test_data), weight, term_index, label_index)
# print("Current label and previous label, f1_score: %f" % (f1))

# phi_1 + phi_2 + phi_3
