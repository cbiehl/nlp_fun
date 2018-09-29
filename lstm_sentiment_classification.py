import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import movie_reviews, stopwords
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import os
import re
import glob
import timeit
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

def tokenize(textstring):
    return nltk.word_tokenize(textstring)

def remove_stopwords(tokens, stopwords=stopwords.words('english')):
    return [t for t in tokens if t not in stopwords]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def pad_zeros(sentence):
    if len(sentence) == NUM_WORDS:
        return sentence
    else:
        return [0] * (NUM_WORDS - len(sentence)) + sentence


def get_input_fn(x, y=None, batch_size=100, num_epochs=1, shuffle=False):
    return tf.estimator.inputs.numpy_input_fn(x={"x": x},
                                              y=y,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              shuffle=shuffle)


def get_avg_sentence_length(sentences):
    return int(sum(len(sentence) for sentence in sentences) / len(sentences))

def get_max_sentence_length(sentences):
    return max(len(sentence) for sentence in sentences)


# model definition
def lstm_model_fn(features, labels, mode, params):

    # Define input layer
    input_layer = features["x"]

    # Embedding layer
    word_embeddings = tf.get_variable(name="word_embeddings",
                                      shape=[params["vocabulary_size"], params["embedding_size"]],
                                      initializer=tf.random_normal_initializer())
    input_layer = tf.nn.embedding_lookup(word_embeddings, input_layer)

    # LSTM (with dropout)
    basic_lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=params["num_units"],
                                                     activation=tf.nn.tanh)
                        for _ in range(params["num_layers"])]
    dropout_lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(basic_lstm_cell, output_keep_prob=params["dropout_prob"])
                          for basic_lstm_cell in basic_lstm_cells]
    multi_lstm_cells = tf.nn.rnn_cell.MultiRNNCell(dropout_lstm_cells)
    outputs, states = tf.nn.dynamic_rnn(multi_lstm_cells, input_layer, dtype=tf.float32)

    # Extract final state (last hidden state of sequence of topmost layer)
    final_state = states[-1].h

    # Fully connected layer (with linear activation)
    logits = tf.squeeze(tf.layers.dense(inputs=final_state, units=1, activation=None))

    # Define output
    sentiment = tf.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"sentiment": tf.round(sentiment)})
    
    # Cast labels
    labels = tf.cast(labels, dtype=tf.float32)

    # Define loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
    
    with tf.name_scope("summaries"):
        tf.summary.scalar("cross_entropy", loss)
        
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            predictions={"sentiment": sentiment},
            eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.cast(tf.round(sentiment), tf.float32))})

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
    
def read_data(path):
    files = sorted(glob.glob(path + "/*.txt"))
    
    for file in files:
        for line in open(file, 'r', encoding="utf-8", errors="ignore"):
            yield line


# Parameters
NUM_WORDS = 400 #'max' #'avg'
TRAIN_RATIO = .9
NUM_EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
NUM_LAYERS = 4
NUM_UNITS = 20
DROPOUT_PROB = .9
EMBEDDING_SIZE = 50
MODEL_PATH = "dump/"
EVERY_N_ITER = 100
CLEAN = True

# make directory
if CLEAN:
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
        os.mkdir(MODEL_PATH)

# load data
path = ""
sentences_pos = list(read_data(path + 'aclImdb/train/pos'))
sentences_pos = [remove_stopwords(tokenize(text)) for text in sentences_pos]
sentences_neg = list(read_data(path + 'aclImdb/train/neg'))
sentences_pos = [remove_stopwords(tokenize(text)) for text in sentences_neg]

#sentences_pos = [list(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('pos')]
#sentences_neg = [list(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('neg')]

if NUM_WORDS == 'avg':
    NUM_WORDS = get_avg_sentence_length(sentences_pos + sentences_neg)
    print("Set NUM_WORDS to:", NUM_WORDS)
    
if NUM_WORDS == 'max':
    NUM_WORDS = get_max_sentence_length(sentences_pos + sentences_neg)
    print("Set NUM_WORDS to:", NUM_WORDS)

# truncate num of words
sentences_pos = [sentence[:NUM_WORDS] for sentence in sentences_pos]
sentences_neg = [sentence[:NUM_WORDS] for sentence in sentences_neg]

# Build dictionary (0 is used for padding)
sentences = sentences_pos + sentences_neg
dictionary = [word for sentence in sentences for word in sentence]
dictionary = list(set(dictionary))
dictionary = dict(zip(dictionary, range(1, len(dictionary) + 1)))

# Map words to integers
sentences_pos = [[dictionary[word] for word in sentence] for sentence in sentences_pos]
sentences_neg = [[dictionary[word] for word in sentence] for sentence in sentences_neg]

# check consistency
#dictionary_inv = {b: a for a, b in dictionary.items()}
#print("POS: " + " ".join([dictionary_inv[index] for index in sentences_pos[0]]))
#print("NEG: " + " ".join([dictionary_inv[index] for index in sentences_neg[0]]))

sentences_pos = [pad_zeros(sentence) for sentence in sentences_pos]
sentences_neg = [pad_zeros(sentence) for sentence in sentences_neg]

data_pos = np.array(sentences_pos, dtype=np.int32)
data_pos_labels = np.ones(shape=[len(sentences_pos)], dtype=np.int32)

data_neg = np.array(sentences_neg, dtype=np.int32)
data_neg_labels = np.zeros(shape=[len(sentences_neg)], dtype=np.int32)

data = np.vstack((data_pos, data_neg))
data_labels = np.concatenate((data_pos_labels, data_neg_labels))

# train/test split
train = []
train_labels = []
test = []
test_labels = []

sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_RATIO, random_state=None)

for train_indices, test_indices in sss.split(data, data_labels):
    for idx in train_indices:
        train.append(data[idx][:])
        train_labels.append(data_labels[idx])
        
    for idx in test_indices:
        test.append(data[idx][:])
        test_labels.append(data_labels[idx])
        

train.extend(test) #TODO!!!       
train_labels.extend(test_labels) #TODO!!!

train = np.array(train, dtype=np.int32)
train_labels = np.array(train_labels, dtype=np.int32)
test = np.array(test, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

#np.random.shuffle(data)
#num_rows = data.shape[0]
#
#split_train = int(num_rows * TRAIN_RATIO)
#train, train_labels = data[:split_train, :], data_labels[:split_train]
#test, test_labels = data[split_train:, :], data_labels[split_train:]

model_params = {"learning_rate": LEARNING_RATE,
                "num_layers": NUM_LAYERS,
                "num_units": NUM_UNITS,
                "embedding_size": EMBEDDING_SIZE,
                "dropout_prob": DROPOUT_PROB,
                "vocabulary_size": len(dictionary) + 1}

loss_hook = tf.train.LoggingTensorHook(["loss"], every_n_iter=EVERY_N_ITER)

# instantiate model
clf = tf.estimator.Estimator(model_fn=lstm_model_fn,
                            params=model_params,
                            model_dir=MODEL_PATH)

# start training
print("Starting model training")
print("Parameters:")
print(model_params)
start = timeit.default_timer()
clf.train(input_fn=get_input_fn(x=train,
                               y=train_labels,
                               batch_size=BATCH_SIZE,
                               num_epochs=NUM_EPOCHS,
                               shuffle=True),
         hooks=[loss_hook])

stop = timeit.default_timer()
print()
print("Finished training")
print("Training time:", stop - start)

# test
eval_dict = clf.evaluate(input_fn=get_input_fn(x=test,
                                               y=test_labels,
                                               batch_size=test.shape[0]))

print("Cross entropy (test set): {0:.6f}".format(eval_dict["loss"]))
print("Accuracy (test set): {0:.6f}".format(eval_dict["accuracy"]))

# evaluate
prediction = clf.predict(input_fn=get_input_fn(x=test,
                                               y=test_labels,
                                               batch_size=test.shape[0]))

predicted_labels = np.array([p["sentiment"] for p in prediction])

#le = preprocessing.LabelEncoder()
#le.fit(test_labels)
#y_true = le.transform(test_labels)
#y_pred = list(prediction)
#y_pred = le.transform(y_pred)
#class_names = le.classes_

cnf_mtrx = confusion_matrix(test_labels, predicted_labels)

fig = plt.figure(figsize=(10,10))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_mtrx, classes=list(set(predicted_labels)),
                      title='Confusion matrix, without normalization')

#%%
import re

digits = re.compile(r'(\d+)')
def digit_order(filename):
    return tuple(int(t) if match else t
                 for t, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))

def read_data(path):
    files = sorted(glob.glob(path + "/*.txt"), key=digit_order)
    
    for file in files:
        for line in open(file, 'r', encoding="utf-8", errors="ignore"):
            yield line

X_test = read_data(path + "data")
X_test = [remove_stopwords(tokenize(text)) for text in X_test]
X_test = [[word for word in sentence if word in dictionary] for sentence in X_test]

# truncate num of words
X_test = [sentence[:NUM_WORDS] for sentence in X_test]

# Map words to integers
X_test = [[dictionary[word] for word in sentence] for sentence in X_test]

X_test = [pad_zeros(sentence) for sentence in X_test]

X_test = np.array(X_test, dtype=np.int32)

prediction = clf.predict(input_fn=get_input_fn(x=X_test,
                                               y=None,
                                               batch_size=X_test.shape[0]))

predicted_labels = [p["sentiment"] for p in prediction]
print(predicted_labels[:10])
predicted_labels = ["pos" if p == 1.0 else "neg" for p in predicted_labels]

with open(path + "answer_rnn.txt", "w") as f:
    for i, label in enumerate(predicted_labels):
        f.write(str(i+1) + "_" + label + "\n")

