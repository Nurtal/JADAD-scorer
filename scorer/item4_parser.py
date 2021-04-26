def simple_guess(title, abstract):
    """
    place holder

    - la methode de randomisation a ete decrite
    dans l'article et que cette methode est appropriee

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
