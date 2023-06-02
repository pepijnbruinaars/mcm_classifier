import matplotlib.pyplot as plt
import numpy as np
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier

# Customizable environment variables
n_categories = 10  # Number of categories to be classified
n_variables = 121  # Number of variables in the dataset
mcm_filename_format = "train-images-unlabeled-{}_comms.dat"
data_filename_format = "train-images-unlabeled-{}.dat"


def plot_confusion_matrix(confusion_matrix, title="Confusion matrix", cmap="Blues"):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_categories)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def main():
    print("{:-^50}".format("  MCM-Classifier  "))

    test_data = load_data("INPUT/data/test-images-unlabeled-all-uniform.txt").astype(
        int
    )
    test_labels = load_labels("INPUT/data/test-labels-uniform.txt").astype(int)

    # Step 1: Initialize classifier
    classifier = MCM_Classifier(
        n_categories, n_variables, mcm_filename_format, data_filename_format
    )
    
    # Step 2: Train
    # classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)
    classifier.init()

    # Step 3: Evaluate
    predicted_classes, probs = classifier.evaluate(test_data, test_labels)

    # Step 4: Save classification report and other stats
    report = classifier.get_classification_report(test_labels)
    classifier.save_classification_report("classification_report_1", test_labels)

    # Plot confusion matrix
    # plt.figure()
    # plot_confusion_matrix(classifier.stats["confusion_matrix"])
    # plt.savefig("OUTPUT/confusion_matrix.png")


if __name__ == "__main__":
    main()
