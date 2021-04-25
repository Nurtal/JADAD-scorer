

def first_try(title, abstract):
    """
    """

    ## importation
    ## parameters
    ## preprocess text
    ## hunt specific target
    ## assert if item is present or not
    ## return boolean

    if "Random" in str(abstract) or "random" in str(abstract):
        return 1
    else :
        return 0



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
