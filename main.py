import numpy as np
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier

# Customizable environment variables
n_categories = 10 # Number of categories to be classified
n_variables = 121 # Number of variables in the dataset
mcm_filename_format = "train-images-unlabeled-{}_comms.dat"
data_filename_format = "train-images-unlabeled-{}.dat"

def main():
    print("MCM-BASED CLASSIFIER")
    
    test_data = load_data("INPUT/data/test-images-unlabeled-all-uniform.txt").astype(int)
    test_labels = load_labels("INPUT/data/test-labels-uniform.txt").astype(int)

    # Step 1: Initialize classifier
    classifier = MCM_Classifier(
        n_categories, n_variables,
        mcm_filename_format, data_filename_format
    )
    
    # Step 2: Evaluate
    predicted_classes, probs, acc = classifier.evaluate(test_data, test_labels)
    
    
if __name__ == "__main__":
    main()