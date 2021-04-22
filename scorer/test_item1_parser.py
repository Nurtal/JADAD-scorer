

def run(test_dataset):
    """
    Run test
    """

    ## imortation
    import pandas as pd
    from . import item1_parser

    score = 0
    ## load test dataset
    df = pd.read_csv(test_dataset)
    for index, row in df.iterrows():

        #-> loop over each entry
        title = row['TITLE']
        abstract = row['ABSTRACT']
        present = row['ITEM1']

        ## run item1 parser
        match = item1_parser.first_try(title, abstract)

        ## comparre match & score
        if str(match) == str(present) :
            score += 1
        return score
