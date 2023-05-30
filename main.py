import numpy as np
from src.loaders import load_data
from src.classify import MCM_Classifier

# Customizable environment variables
n_categories = 2 # Number of categories to be classified
n_variables = 121 # Number of variables in the dataset

def main():
    print("MCM-BASED CLASSIFIER")
    
    # Step 1: classify
    classifier = MCM_Classifier(n_categories, n_variables)
    classifier.classify()
    
    # Step 2: evaluate
    
if __name__ == "__main__":
    main()