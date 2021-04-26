

def display_help():
    """
    Display help for the test run
    """

    ## help
    print("""

        Tadam

    """)



## MAIN SCRIPT ##
if __name__=="__main__":

    ## importation
    import scorer
    from scorer import test_item1_parser
    from scorer import test_item2_parser
    from scorer import test_item3_parser
    from scorer import test_item4_parser
    from scorer import test_item5_parser

    ## parameters
    dataset_test_file = "data/test_dataset.csv"
    shutup_mode = False

    ## test item 1 parsing
    i1_results = scorer.test_item1_parser.test_simple_guess(dataset_test_file)

    #-> display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM1] ACC => "+str(i1_results[0]))
        print("[ITEM1] AUC => "+str(i1_results[1]))

    ## test item 2 parsing
    i2_results = scorer.test_item2_parser.test_simple_guess(dataset_test_file)

    #-> display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM2] ACC => "+str(i2_results[0]))
        print("[ITEM2] AUC => "+str(i2_results[1]))

    ## test item 3 parsing
    i3_results = scorer.test_item3_parser.test_simple_guess(dataset_test_file)

    #-> display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM3] ACC => "+str(i3_results[0]))
        print("[ITEM3] AUC => "+str(i3_results[1]))

    ## test item 4 parsing
    i4_results = scorer.test_item4_parser.test_simple_guess(dataset_test_file)

    #-> display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM4] ACC => "+str(i4_results[0]))
        print("[ITEM4] AUC => "+str(i4_results[1]))

    ## test item 5 parsing
    i5_results = scorer.test_item5_parser.test_simple_guess(dataset_test_file)

    #-> display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM5] ACC => "+str(i5_results[0]))
        print("[ITEM5] AUC => "+str(i5_results[1]))
