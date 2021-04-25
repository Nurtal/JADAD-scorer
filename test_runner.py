

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

    ## parameters
    dataset_test_file = "data/test_dataset.csv"
    shutup_mode = False

    ## test item 1 parsing
    i1_results = scorer.test_item1_parser.test_simple_guess(dataset_test_file)

    ## display results if shutup_mode is set to False
    if(not shutup_mode):
        print("[ITEM1] ACC => "+str(i1_results[0]))
        print("[ITEM1] AUC => "+str(i1_results[1]))
