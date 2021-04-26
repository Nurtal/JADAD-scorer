def test_simple_guess(test_dataset):
    """
    """

    ## imortation
    import pandas as pd
    from . import item5_parser
    from sklearn import metrics

    ## parameters
    match_prediction = []
    match_label = []

    ## load test dataset
    df = pd.read_csv(test_dataset)
    for index, row in df.iterrows():

        #-> loop over each entry
        title = row['TITLE']
        abstract = row['ABSTRACT']
        present = row['ITEM5']

        ## update list of labels
        match_label.append(present)

        ## run item1 parser
        match = item5_parser.simple_guess(title, abstract)

        ## update list of preditcion
        if(match):
            match_prediction.append(1)
        else:
            match_prediction.append(0)

    ## compure accuracy
    acc_score = metrics.accuracy_score(match_label, match_prediction)

    ## compute auc
    fpr, tpr, thresholds = metrics.roc_curve(match_label, match_prediction)
    auc_score = metrics.auc(fpr, tpr)

    ## return acc and auc as a tuple
    return(acc_score, auc_score)
