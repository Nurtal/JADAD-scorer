def simple_guess(title, abstract):
    """
    simple & stupid regex search

    - Est-ce que cet article est decrit comme randomise ?
    """

    ## importation
    import re

    ## parameters
    match_item = False

    ## preprocess text
    text = str(title)+". "+str(abstract)
    text = text.replace(".. ", ". ")
    text = text.lower()

    ## hunt random
    if(re.search('random', text)):
        match_item = True

    ## return match status
    return match_item



def smart_guess(title, abstract):
    """
    simple & stupid regex search to locate sentence of interest,
    Biobert vectorisation, compute distance from set of target sentences

    - Est-ce que cet article est decrit comme randomise ?
    """

    ## importation
    import re

    ## parameters
    match_item = False

    ## preprocess text
    text = str(title)+". "+str(abstract)
    text = text.replace(".. ", ". ")
    text = text.lower()

    ## hunt interesting sentences
    sentence_list = text.split(". ")
    sentence_saved = []
    for sentence in sentence_list:
        if(re.search('random', sentence)):
            sentence_saved.append(sentence)

    ## Loop over saved sentences
    for sentence in sentence_saved:

        pass
        ## TODO : vectorize sentence

        ## TODO : loop over vectorized target sentences

        ## TODO : compute distance between target and saved

        ## TODO : if distance below treshold set match_item to True


    ## return match status
    return match_item


def deep_guess(title, abstract):

    """
    craft text from title and abstract, split into sentence, use a pretrained
    RNN on each sentence to predict if sentence contains item 1
    """

    ## importation
    import keras
    import pickle
    from keras.preprocessing import sequence
    from keras.utils import to_categorical

    ## parameters
    match_item = False
    model_filename = "../models/item1_predictor.h5"
    tokenize_filename = "../models/item1_tokenizer.pickle"
    max_len = 250

    ## preprocess text
    text = str(title)+". "+str(abstract)
    text = text.replace(".. ", ". ")
    text = text.lower()
    sentence_list = text.split(". ")

    ## load RNN model
    model = keras.models.load_model(model_filename)

    ## load tokenizer
    with open(tokenize_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)

    ## tokenize sentences
    seq_list = tokenizer.texts_to_sequences(sentence_list)
    sequences_matrix = sequence.pad_sequences(seq_list,maxlen=max_len)

    ## make prediction
    prediction = model.predict(sequences_matrix)
    preditcion = to_categorical(prediction)

    ## check prediction
    for pred in preditcion:
        if(pred[1] == 1.0):
            match_item = True

    ## return match status
    return match_item


deep_guess('tadam', "Methods: A randomized, double-blind, crossover of 4 weeks was conducted. and also fucking murlock")
