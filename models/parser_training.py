


## importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle

## load sentence data
df = pd.read_csv('../data/sentence_dataset.csv')

## init log file
log_file = open("item_parser_training.log", "w")
log_file.write("ITEM,LOSS,ACC,AUC\n")

## model function
def RNN(max_len, max_words):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model



## loop over item to parse
for i in range(1,6):

    ## parameters
    item_name = "ITEM "+str(i)
    model_save_name = "item"+str(i)+"_predictor.h5"
    tokenizer_save_name = "item"+str(i)+"_tokenizer.pickle"
    acc_fig_save_name = "images/acc_item"+str(i)+"_evolution.png"
    auc_fig_save_name = "images/auc_item"+str(i)+"_evolution.png"
    max_words = 5000
    max_len = 250
    epoch_nb = 350
    batch_size = 16

    ## preprocess
    X = df['SENTENCE']
    Y = df[item_name]
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)

    ## split into train & validation
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35)

    ## tokenisation
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    ## EarlyStopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    ## craft model
    model = RNN(max_len,max_words)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy', tf.keras.metrics.AUC()])

    ## train the model
    history = model.fit(
        sequences_matrix,Y_train,
        batch_size=batch_size,
        epochs=epoch_nb,
        validation_split=0.3,
        callbacks=[callback]
    )

    ## save model
    model.save(model_save_name)

    ## save tokenizer
    with open(tokenizer_save_name, 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## evaluate the model
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix,Y_test)
    loss_val = accr[0]
    acc_val = accr[1]
    auc_val = accr[2]

    ## save perf in log file
    log_file.write(item_name+","+str(loss_val)+","+str(acc_val)+","+str(auc_val)+"\n")

    ## save perf evolution graph - accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(acc_fig_save_name)
    plt.close()

    ## save perf evolution graph - auc
    if(i ==1):
        auc_key = 'auc'
    else:
        auc_key = "auc_"+str(i-1)
    plt.plot(history.history[auc_key])
    plt.plot(history.history["val_"+auc_key])
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(auc_fig_save_name)
    plt.close()


## close file
log_file.close()
