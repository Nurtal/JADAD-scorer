def simple_guess(title, abstract):
    """
    simple & stupid regex search

    - Est ce que cet article est decrit comme en double aveugle ?
    (On doit plutôt dire maintenant que le patient,
    le soignant et l'evaluateur sont en aveugle.)
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
    #-> TODO : adapt, just placeholder for now
    if(re.search('blind', text)):
        match_item = True

    ## return match status
    return match_item
