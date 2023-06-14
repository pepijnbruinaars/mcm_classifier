import matplotlib.pyplot as plt
import numpy as np
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier
from src.plot import plot_confusion_matrix, plot_label_prob_diff

# Customizable environment variables
n_categories = 10  # Number of categories to be classified
n_variables = 121  # Number of variables in the dataset
mcm_filename_format = "train-images-unlabeled-{}_comms.dat"
data_filename_format = "train-images-unlabeled-{}.dat"

def main():
    print("{:-^50}".format("  MCM-Classifier  "))

    test_data = load_data("input/data/test-images-unlabeled-all-uniform.txt").astype(
        int
    )
    test_labels = load_labels("input/data/test-labels-uniform.txt").astype(int)

    # Step 1: Initialize classifier
    classifier = MCM_Classifier(
        n_categories, n_variables, mcm_filename_format, data_filename_format
    )

    # Step 2: Train
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)
    # classifier.init()

    # Step 3: Evaluate
    predicted_classes, probs = classifier.evaluate(test_data, test_labels)

    # Step 4: Save classification report and other stats
    report = classifier.get_classification_report(test_labels)
    classifier.save_classification_report(test_labels)

    if (classifier.stats == None):
        raise Exception("Classifier stats not found. Did you forget to call evaluate()?")

    # Count amount of -1 labels
    n_no_labels = 0
    no_labels_labels = []
    images = []
    for i in range(len(test_labels)):
        if predicted_classes[i] == -1:
            n_no_labels += 1
            no_labels_labels.append(test_labels[i])
            images.append(test_data[i])

    # Plot all images with no labels so that they are in the same figure in a square grid
    dim_1 = int(np.sqrt(n_no_labels))
    dim_2 = int(np.sqrt(n_no_labels))
    fig, axs = plt.subplots(dim_1, dim_2, figsize=(10, 10))
    for i in range(dim_1):
        for j in range(dim_2):
            axs[i, j].imshow(images[i*dim_2 + j].reshape(11, 11), cmap="gray")
            axs[i, j].set_title(no_labels_labels[i*dim_2 + j])
            axs[i, j].axis("off")
    
    plt.show()
            
    print("Number of datapoints for which the classifier didn't have any probability for any category: {}".format(n_no_labels))
    print("Labels of these datapoints: {}".format(no_labels_labels))        
    
    # Find amount of datapoints with 2 or more categories with probability > 0
    n_multiple_probs = 0
    multiple_probs_labels = []
    for i in range(len(probs)):
        if np.sum(probs[i] > 0) > 1:
            n_multiple_probs += 1
            multiple_probs_labels.append(test_labels[i])
    print("Number of datapoints with 2 or more categories with probability > 0: {}".format(n_multiple_probs))

    plot_label_prob_diff(3, 5, test_labels, probs, predicted_classes)
    # plt.savefig("OUTPUT/probs_and_correctness.png")

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(classifier.stats["confusion_matrix"], n_categories)
    plt.savefig("OUTPUT/confusion_matrix.png")


if __name__ == "__main__":
    main()
