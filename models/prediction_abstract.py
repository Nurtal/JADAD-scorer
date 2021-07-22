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

## parameters
max_words = 5000
max_len = 250


def item_predition(max_words,max_len,sentence):
    info_abstract = {}
    for i in range(1,6):
        save_name_model = 'item'+str(i)+'_predictor.h5'
        save_name_tokenizer = 'item'+str(i)+'_tokenizer.pickle'

        model = tf.keras.models.load_model(str(save_name_model))
        with open(str(save_name_tokenizer),'rb') as handle :
            loaded_tokenizer = pickle.load(handle)

        sequences = loaded_tokenizer.texts_to_sequences([sentence])
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
        pred = model.predict(sequences_matrix)
        print("ITEM" +str(i)+" : "+str(pred))
        prediction_value = pred[0][0]
        new_pred = '{:.20f}'.format(prediction_value)
        new_pred = round(float(new_pred),3)
        print(new_pred)
        if new_pred >= 0.001 :
            if str("ITEM")+str(i) not in info_abstract.keys():
                info_abstract[str("ITEM")+str(i)] = [str(sentence)]
            else :
                info_abstract[str("ITEM")+str(i)].append(str(sentence))

    print(info_abstract)



sentence = "All potential events were adjudicated by a clinical events committee, which was blinded to treatment allocation."
item_predition(max_words,max_len,sentence)

"""
ESSAIS :
- "Participants were randomized into one of two groups upon enrollment using a 1:1 allocation ratio." : item 1 et 4
- "Effects of melatonin supplementation on serum oxidative stress markers and disease activity in systemic lupus erythematosus patients: A randomised, double-blind, placebo-controlled trial" : item 1 et 2
- "Briefly, 14 men and 13 women completed the study and one man was excluded from these analyses since he was considered an outlier based on his food intake data." : item 3
- "After being informed, 46 patients, 36 women and 10 men, refused further cooperation." : item 3
- "Four patients were excluded for decortication operation (n = 2) and postoperative cooperation problem (n = 2)." : item 3
- "Randomization was performed with the method of closed envelopes selected by patients just before the surgery." : pas item 4 (0.00003)
- "After completion of the screening, 6 participants were excluded (3 changed their mind, 1 became pregnant, 1 had sleep duration > 6 h/night according to actigraphy and 1 could not tolerate glucose solution during an OGTT), and 22 were then randomized" : item 1 pas 3
- "The data collectors were all blinded to the intervention conditions." : item 5
- "All potential events were adjudicated by a clinical events committee, which was blinded to treatment allocation." : item 5
"""
