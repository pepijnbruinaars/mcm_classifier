import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier
from src.plot import plot_confusion_matrix, plot_label_prob_diff

# Customizable environment variables
n_categories = 10  # Number of categories to be classified
n_variables = 121  # Number of variables in the dataset
mcm_filename_format = "train-images-unlabeled-{}_bootstrap_comms.dat"
data_filename_format = "train-images-unlabeled-{}_bootstrap.dat"

def main():
    # Print current working directory
    print("Current working directory: {}".format(os.getcwd()))
    test_data = load_data("input/data/test-images-unlabeled-all-uniform.txt").astype(
        int
    )
    test_labels = load_labels("input/data/test-labels-uniform.txt").astype(int)
    
    resulting_data = pd.DataFrame(
        # columns=["sample_size", "run", "fit-time", "acc", "avg-prec", "avg-rec", "avg-f1", "prec-0", "prec-1", "prec-2", "prec-3", "prec-4", "prec-5", "prec-6", "prec-7", "prec-8", "prec-9", "rec-0", "rec-1", "rec-2", "rec-3", "rec-4", "rec-5", "rec-6", "rec-7", "rec-8", "rec-9", "f1-0", "f1-1", "f1-2", "f1-3", "f1-4", "f1-5", "f1-6", "f1-7", "f1-8", "f1-9"],
        # dtype=float
        )
    
    sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 0]
    progress_bar = tqdm(total=len(sample_sizes) * 5)
    for sample_size in sample_sizes:
        for i in range(5):
            print("Sample size: {}".format(sample_size))
            # Step 1: Initialize classifier
            classifier = MCM_Classifier(
                n_categories, n_variables, mcm_filename_format, data_filename_format
            )
        
            # Step 2: Train
            # Time the fitting
            start_time = time.perf_counter()
            classifier.fit(greedy=True, n_samples=sample_size, max_iter=1000000, max_no_improvement=100000)
            # classifier.init()
            end_time = time.perf_counter()
        
            # Step 3: Evaluate
            predicted_classes, probs = classifier.evaluate(test_data, test_labels)

            # Step 4: Save classification report and other stats
            report = classifier.get_classification_report(test_labels)
            classifier.save_classification_report(test_labels)
        
            if (classifier.stats == None):
                raise Exception("Classifier stats not found. Did you forget to call evaluate()?")
            
            # Append data to resulting_data
            resulting_data = resulting_data._append(pd.Series({
                "sample_size": sample_size,
                "run": i + 1,
                "fit-time": end_time - start_time,
                "n_rejected": report["rejected"],
                "true_accuracy": report["true_accuracy"],
                "non_rejected_accuracy": report["non_rejected_accuracy"],
                "classification_quality": report["classification_quality"],
                "avg-prec": report["avg_precision"],
                "avg-rec": report["avg_recall"],
                "avg-f1": report["avg_f1_score"],
                "prec-0": report["precision"][0],
                "prec-1": report["precision"][1],
                "prec-2": report["precision"][2],
                "prec-3": report["precision"][3],
                "prec-4": report["precision"][4],
                "prec-5": report["precision"][5],
                "prec-6": report["precision"][6],
                "prec-7": report["precision"][7],
                "prec-8": report["precision"][8],
                "prec-9": report["precision"][9],
                "rec-0": report["recall"][0],
                "rec-1": report["recall"][1],
                "rec-2": report["recall"][2],
                "rec-3": report["recall"][3],
                "rec-4": report["recall"][4],
                "rec-5": report["recall"][5],
                "rec-6": report["recall"][6],
                "rec-7": report["recall"][7],
                "rec-8": report["recall"][8],
                "rec-9": report["recall"][9],
                "f1-0": report["f1_score"][0],
                "f1-1": report["f1_score"][1],
                "f1-2": report["f1_score"][2],
                "f1-3": report["f1_score"][3],
                "f1-4": report["f1_score"][4],
                "f1-5": report["f1_score"][5],
                "f1-6": report["f1_score"][6],
                "f1-7": report["f1_score"][7],
                "f1-8": report["f1_score"][8],
                "f1-9": report["f1_score"][9],
            }), ignore_index=True) # type: ignore
            # print true acc, non-rej acc, class qual
            print(resulting_data)
            progress_bar.update(1)
            
    # Save resulting_data to csv
    resulting_data.to_csv("output/data/bootstrap_results_greedy.csv", index=False, sep=";")
    

    
if __name__ == "__main__":
    main()