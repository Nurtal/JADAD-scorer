def simple_guess(title, abstract):
    """
    simple & stupid regex search
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
