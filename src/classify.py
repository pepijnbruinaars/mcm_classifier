from .helpers import *

def classify(n_categories: int) -> None:
    """
    Classify the data using the MCM-based classifier.

    Args:
        n_categories (int): The number of categories in the dataset
    """
    P = [] # Probability distributions for each mcm
    MCM = [] # Communities for each mcm
    
    construct_P(P, MCM, n_categories)
    
    return None