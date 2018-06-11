import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from data import load_wiki_data, load_legal_data, load_glove
from model import Model


def shuffleSplit(ss, X, y, groups):
    #encompass both group and stratified shuffle splitting
    if groups is not None:
        return ss.split(X, y, groups)
    else:
        return ss.split(X, y)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Use 'legal' (default) or 'wiki' dataset", choices=['legal', 'wiki'], default='legal')
parser.add_argument("--baseline", help="Use baseline classifier", action="store_true")
parser.add_argument("-s", "--splits", help="Number of cross-validations", type=int, default=1)
parser.add_argument("-t", "--test_size", help="Proportion of test split", type=float, default=0.2)
parser.add_argument("-g", "--glove_size", help="GloVe embedding size", type=int, choices=[50, 100, 200, 300], default=50)
parser.add_argument("-n", "--hidden_size", help="RNN hidden weights size", type=int, default=8)
parser.add_argument("-l", "--lr", help="RNN learning rate", type=float, default=0.01)
parser.add_argument("-r", "--dropout", help="RNN dropout rate", type=float, default=0.1)
parser.add_argument("-c", "--cell_type", help="RNN cell type; 'g' for GRU (default) or 'l' for LSTM", choices=['g', 'l'], default='g')
parser.add_argument("-b", "--batch_size", help="Training batch size", type=int, default=512)
parser.add_argument("-e", "--epochs", help="Training epochs", type=int, default=2)
parser.add_argument("-o", "--random_seed", help="Global random seed (cross-validation split, batch shuffling and Tensorflow initializations and operations)", type=int, default=None)
parser.add_argument("-v", "--visualize", help="Visualize output as HTML; use 's' for 0.1%% samples or 'e' for all errors", choices=[None, 's', 'e'], default=None)
args = parser.parse_args()

if args.dataset == 'wiki':
    #train_set, dev_set, test_set = load_wiki_data()
    X, y, groups = load_wiki_data(mode='cv')
    ss = GroupShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.random_seed)
elif args.dataset == 'legal':
    #train_set, dev_set, test_set = load_legal_data()
    X, y = load_legal_data(mode='cv')
    groups = None
    ss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.random_seed)

if args.random_seed is not None:
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

data = pd.concat([X, y], axis=1)

word2id, embed_matrix = load_glove(args.glove_size)

if args.cell_type == 'l':
    cell_type = tf.nn.rnn_cell.LSTMCell
else:
    cell_type = tf.nn.rnn_cell.GRUCell

model = Model(word2id, embed_matrix, args.hidden_size, args.lr, args.dropout, cell_type, args.batch_size, args.epochs, seed=args.random_seed)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True

scores = []
for train, test in shuffleSplit(ss, X, y, groups):
    train_data = data.iloc[train]
    test_data = data.iloc[test]
    #test_data = pd.DataFrame([["thanks for watching !", 0]]) #evaluate arbitrary sentence, for fun
    if args.baseline:
        model = SVC(kernel='linear')
        #model = GaussianNB() #faster classifier for testing
        train_features = train_data[0].apply(lambda x: len(str(x).split())).to_frame()
        train_features = pd.concat([train_features, train_data[0].apply(lambda x: len(str(x))).to_frame()], axis=1)
        print("Fitting...")
        model.fit(train_features, train_data[1])
        test_features = test_data[0].apply(lambda x: len(str(x).split())).to_frame()
        test_features = pd.concat([test_features, test_data[0].apply(lambda x: len(str(x))).to_frame()], axis=1)
        print("Predicting...")
        preds = model.predict(test_features)
        acc = accuracy_score(test_data[1], preds)
    else:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.fit(sess, train_data, batch_random_seed=args.random_seed)
            acc = model.predict(sess, test_data, viz=args.visualize, batch_random_seed=args.random_seed)
    scores.append(acc)
    print(np.mean(scores))
