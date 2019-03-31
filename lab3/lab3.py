import  argparse, time
import numpy as np
import matplotlib.pyplot as plt

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
    f1_scores = []
    for i in range(iterate_num):
        start = time.clock()
        np.random.seed(i)
        train_data = np.random.permutation(train_data)
        for j in range(len(train_data)):
            prediction = predict(weight, train_data[j][0], labels, keys, phi_n)
            update = not prediction[1] == train_data[j][1]
            if update:
                weight = update_weight(weight, train_data[j], prediction, keys, phi_n)
            # break
        weight_sum += weight  # sum

        print('Cost of %d th iteration: %.2fs ' % ( i+1 ,(time.clock() - start)))
        if(valid_func):
            f1_score = valid_func(weight, labels, keys, phi_n)
            print("f1_score: %f \n" % f1_score)
            f1_scores.append(f1_score)

        start = time.clock()

    return weight_sum / iterate_num, f1_scores


def update_weight(weight, corrected, predicted, keys, phi_n):
    for phi in phi_n:
        co_dict = phi(corrected[0],corrected[1])
        pr_dict = phi(predicted[0],predicted[1])
        for k in co_dict:
            weight[keys[k]] += co_dict[k]
        for k in pr_dict:
            weight[keys[k]] -= pr_dict[k]
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
    for w in terms:
        term_labels.append([(w, l) for l in  labels])

    iterate = product(*list(term_labels))

    for pro_term in iterate:
        c_w = [t[0] for t in pro_term]  # current word, current label
        c_l = [t[1] for t in pro_term]
        permutation_s_t.append((tuple(c_w), tuple(c_l)))

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
        return f1_score(correct, predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])

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
        l.append(y[i-1] + "&" +y[i] +"_" + y[i])

    return dict(Counter(l))


# suffix-3 and current label
def phi_3(x,y):
    l = []
    for  i in range(len(x)) :
        if len(x[i]) > 3:
            l.append(x[i][-3:] + "_" + y[i])
    return dict(Counter(l))


# current word and previous word
def phi_4(x,y):
    l = []
    x = list(x)
    x.insert(0, "<s>")

    for i in range(1,len(x)):
        l.append(x[i] + "&" +x[i-1] + "_" +  y[i-1])

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
        keys.extend([_+"&"+ l + "_" + l for _ in labels])
        keys.append( "NULL&" + l +"_"+ l)

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
def get_keys_for_phi4(data):
    cw_pw = []
    sentences = []
    keys = []
    labels = []
    for s in data:
        sentence = ["<s>"]
        for term in s:
            sentence.append(term[0])
            labels.append(term[1])
        sentences.append(sentence)

    for s in sentences:
        for i in range(1, len(s)):
            cw_pw.append(s[i] + "&" + s[i-1])

    cw_pw = np.unique(cw_pw)
    labels = np.unique(labels)

    for term in cw_pw:
        for  l in labels:
            keys.append(term + "_" + l)

    return keys


# to combine all the keys passed in
def combine_keys(ks):
    keys = []
    for k in ks:
        keys.extend(k)
    keys = np.unique(keys)
    key_dict ={}# dict([(k,v) for (k,v) in zip(keys, range(len(keys)))])
    for (k, v) in zip(keys, range(len(keys))):
        key_dict[k] = v

    return  key_dict


# get the labels in the data set
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


# plot the score and compare
def plot_fs(f1_scores, labels):
    for i in range(len(f1_scores)):
        plt.plot(range(1,len(f1_scores[i])+1), f1_scores[i], label = labels[i])
    plt.xlabel('i th iteration')
    plt.ylabel('f1 score')
    plt.title("F1 Score trend in different feature set")
    plt.legend()
    plt.show()


# print the top ten term in tag
def print_top_ten(weight, keys, labels):
    cls = dict([(label,[]) for label in labels])

    for k in keys:
        label = k.split("_")[1]
        if label in cls:
            cls[label].append((weight[keys[k]], k))

    # sort
    for c in cls:
        cls[c].sort(key = lambda e:e[0], reverse=True)

    # print
    for c in cls:
        topTen = cls[c][:10]
        print("Label ----%s---- most positively-weighted features:"  % c)
        for e in topTen:
            print("Term: %s ,  weight: %f" %(e[1].split("_")[0], e[0]))
        print("\n")
    print("\n\n")


#------------------- main ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('train_file', type=str, help="Training  file")
parser.add_argument('test_file', type=str, help="Testing file")
args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file


# load data
train_data = load_dataset_sents(train_file)
test_data = load_dataset_sents(test_file)

# set random seed
ITERATE_NUM = 15
f1score = []
flabels = []
labels = get_labels(train_data)

_start = 0
# phi_1
print("Only phi_1 ------------------- ")
phi_1_keys = get_keys_for_phi1(train_data)
keys = combine_keys([phi_1_keys])
weight = np.zeros((len(keys.keys()), 1)) # to give it a initial value
weight, f1 = train(get_format_data(train_data), weight, keys, labels,ITERATE_NUM, [phi_1], valid_func = valid_data(get_format_data(test_data)))
print('Cost : %.2fs ' % (time.clock() - _start))
print(f1)
f1score.append(f1)
flabels.append("phi1")
print_top_ten(weight, keys, labels)

#
# # phi_1 + phi_2
# print("Combine phi_1 and phi_2------------------- ")
# phi_1_keys = get_keys_for_phi1(train_data)
# phi_2_keys = get_keys_for_phi2(train_data)
# keys = combine_keys([phi_1_keys, phi_2_keys])
# weight = np.zeros((len(keys.keys()), 1)) # to give it a initial value
# weight,f2 = train(get_format_data(train_data), weight, keys, labels,ITERATE_NUM, [phi_1, phi_2], valid_func = valid_data(get_format_data(test_data)))
# f1score.append(f2)
# flabels.append("phi1+phi2")
# print_top_ten(weight, keys, labels)
#
#
# # phi_1 + phi_2 + phi_3 + phi_4
# print("Combine phi_1 + phi_2 + phi_3 + phi_4------------------- ")
# phi_1_keys = get_keys_for_phi1(train_data)
# phi_2_keys = get_keys_for_phi2(train_data)
# phi_3_keys = get_keys_for_phi3(train_data)
# phi_4_keys = get_keys_for_phi4(train_data)
# keys = combine_keys([phi_1_keys, phi_2_keys, phi_3_keys, phi_4_keys])
# weight = np.zeros((len(keys.keys()), 1)) # to give it a initial value
# weight,f3 = train(get_format_data(train_data), weight, keys, labels,ITERATE_NUM, [phi_1, phi_2, phi_3, phi_4], valid_func = valid_data(get_format_data(test_data)))
# f1score.append(f3)
# flabels.append("phi1+phi2+phi3+phi4")
# print_top_ten(weight, keys, labels)
#
# plot_fs(f1score,flabels)
