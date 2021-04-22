

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

    ## test item 1 parsing
    scorer.test_item1_parser.run(dataset_test_file)
