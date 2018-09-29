import numpy as np
import math
from sklearn.preprocessing import LabelBinarizer
import nltk
import keras
from keras import backend as K
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, \
                         Dense, Dropout, LSTM, Bidirectional, Flatten, \
                         TimeDistributed
import tensorflow as tf

#constants
DATA                = "moviereviews.csv"
EMBEDDINGS          = "glove.6B.100d.txt"
TEST_RATIO          = 0.2
EMBEDDING_DIM       = 100
MAX_SEQUENCE_LENGTH = 100
N_EPOCHS            = 20
BATCH_SIZE          = 100
OOV_TOKEN           = "<<OOV_TOKEN>>"
PADDING_TOKEN       = "<<PADDING>>"


def read_data(path, labelencoder=None, separator=','):
    x = []
    y = []
    labels = set()

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            # remove trailing spaces, newlines, ...
            line = line.strip()

            # create a new document if empty string
            if line == "":
                continue

            # split into tokens
            # tokens[0] => document
            # tokens[1] => label
            splitline = line.split(separator)
            x.append(nltk.word_tokenize(splitline[0]))
            y.append(splitline[1])
            labels.add(splitline[1])

    if labelencoder is None:
        labelencoder = LabelBinarizer()
        y_one_hot = labelencoder.fit_transform(y)
    else:
        y_one_hot = labelencoder.transform(y)

    return np.asarray(x), y_one_hot, labelencoder


def read_embeddings(file_path, embedding_dim, index_dict=None, vector_dict=None):
    """
    Reads the embeddings.

    :param file_path:       path to embedding file
    :param embedding_dim:
    :index_dict:
    :vector_dict:
    :return:                index dictionary, embedding matrix
    """

    if index_dict == None or vector_dict == None:
        index_dict = dict()
        vector_dict = dict()

    with open(file_path, "r", encoding="utf8") as f:
        i = 0
        for line in f:
            split_line = line.strip().split()
            if len(split_line) != embedding_dim + 1:
                continue

            word = split_line[0]

            if word not in index_dict:
                index_dict[word] = i
                vector_dict[word] = np.asarray([float(s) for s in split_line[1:]])
                i += 1
            else:
                # average word vectors across languages
                vector_dict[word] = (vector_dict[word] + np.asarray([float(s)
                                                                     for s in split_line[1:]])) / 2

    # add padding token and oov token
    index_dict[PADDING_TOKEN] = i
    vector_dict[PADDING_TOKEN] = np.zeros(len(split_line) - 1)
    index_dict[OOV_TOKEN] = i + 1
    vector_dict[OOV_TOKEN] = np.random.uniform(-1, 1, size=len(split_line) - 1)

    return index_dict, vector_dict


def f1(y_true, y_pred):
    """
    Computes F1 score.

    :param y_true:
    :param y_pred:
    :return:
    """

    def recall(y_true, y_pred):
        """
        Computes the recall metric.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.

        :param y_true:
        :param y_pred:
        :return:
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Computes the precision metric.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.

        :param y_true:
        :param y_pred:
        :return:
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def embed(raw_seq, index_dict):
    """
    Gets word vector indices for all word sequences.

    :param raw_seq: raw sequence of string tokens
    :param index_dict: dictionary for embedding integer indices
    :return: numpy array containing embedding integer index for each token
    """
    return np.asarray([index_dict[word.lower()]
                       if word.lower() in index_dict
                       else index_dict[OOV_TOKEN] for word in raw_seq])


def export_model(model, path):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'document': model.input}, outputs={'scores': model.output})

    builder = tf.saved_model.builder.SavedModelBuilder('/tmp/saved_model')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        })
    builder.save()


def build_model(embeddings_path, output_dim):
    index_dict, vector_dict = read_embeddings(embeddings_path, EMBEDDING_DIM)
    model = Sequential()

    embedding_weights = np.zeros((len(index_dict), EMBEDDING_DIM))
    for word, index in index_dict.items():
        embedding_weights[index, :] = vector_dict[word]

    # define inputs here
    embedding = Embedding(
        output_dim=EMBEDDING_DIM, input_dim=len(index_dict),
        input_length=MAX_SEQUENCE_LENGTH, trainable=False
        )
    embedding.build((None,))
    embedding.set_weights([embedding_weights])

    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedding_layer = embedding(input_layer)

    # add layers
    model.add(embedding)
    model.add(Dropout(0.4))
    model.add(Conv1D(filters=10, kernel_size=5, padding="same"))
    model.add(keras.layers.PReLU())
    model.add(MaxPooling1D(pool_size=3))
    model.add(Bidirectional(LSTM(50, recurrent_dropout=0.4, return_sequences=True)))
    model.add(Bidirectional(LSTM(50, recurrent_dropout=0.4, return_sequences=False)))
    model.add(Dense(output_dim, activation="softmax"))

    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=[f1])

    return model, index_dict


def train(model, batch_size, x_train_emb, x_test_emb, y_train, y_test):
    print("Starting model training")
    save_path = "keras_model.hdf5"

    earlystop = callbacks.EarlyStopping(monitor="val_f1", min_delta=0.001, patience=2, verbose=0, mode="max")
    modelcheck = keras.callbacks.ModelCheckpoint(save_path, monitor="val_f1", verbose=2, save_best_only=True,
                                                 mode="max")

    model.fit(
        x_train_emb, y_train,
        epochs=20, batch_size=batch_size,
        validation_data=[x_test_emb, y_test],
        callbacks=[modelcheck, earlystop]
    )

    return model


def shuffle(a, b):
    random_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)

    return a, b


# run it :-)
model, index_dict = build_model(EMBEDDINGS, 1)
print(model.summary())
print()
print("Loading data")
x, y, labelencoder = read_data(DATA)
x, y = shuffle(x, y)
x = keras.preprocessing.sequence.pad_sequences(x,
                                               maxlen=MAX_SEQUENCE_LENGTH,
                                               dtype='str',
                                               padding='post',
                                               truncating='post',
                                               value=index_dict[PADDING_TOKEN])
x_train = x[:math.ceil(x.shape[0] * (1 - TEST_RATIO))]
x_test = x[math.floor(x.shape[0] * TEST_RATIO):]
y_train = y[:math.ceil(x.shape[0] * (1 - TEST_RATIO))]
y_test = y[math.floor(x.shape[0] * TEST_RATIO):]
x_train_emb = np.asarray([embed(doc, index_dict) for doc in x_train])
x_test_emb = np.asarray([embed(doc, index_dict) for doc in x_test])

model = train(model, BATCH_SIZE, x_train_emb, x_test_emb, y_train, y_test)

test_example = embed(nltk.word_tokenize("This movie was really horrible!"), index_dict)
print(model.predict(test_example))

export_model(model, "cnn_bilstm")
