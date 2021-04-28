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

        ## TODO : vectorize sentence

        ## TODO : loop over vectorized target sentences

        ## TODO : compute distance between target and saved

        ## TODO : if distance below treshold set match_item to True


    ## return match status
    return match_item
