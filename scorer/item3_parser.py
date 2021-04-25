def simple_guess(title, abstract):
    """
    place holder

    - Existe-t-il une description des retraits
    d'etudes et des abandons ?
    """

    ## importation
    import re

    ## parameters
    match_item = False

    ## preprocess text
    text = str(title)+". "+str(abstract)
    text = text.replace(".. ", ". ")
    text = text.lower()

    ## TODO : Magick

    ## return match status
    return match_item
