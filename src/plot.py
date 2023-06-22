import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def plot_confusion_matrix(confusion_matrix, n_categories: int, title="Confusion matrix", cmap="Blues", logScale: bool = False):
    """
    This function prints and plots the confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        n_categories (int): Number of categories
        title (str, optional): Title of the plot. Defaults to "Confusion matrix".
        cmap (str, optional): Color map. Defaults to "Blues".
    """
    if logScale:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap, norm=LogNorm(vmin=1, vmax=confusion_matrix.max()))
    else:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_categories)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
def plot_label_prob_diff(label1, label2, test_labels, probs, predicted_classes, title="Label probability difference"):
    correctly_classified_as_label1 = []
    correctly_classified_as_label2 = []
    incorrectly_classified_as_label1 = []
    incorrectly_classified_as_label2 = []
    for i in range(len(probs)):
        # Correctly classified as category 'label1'
        if test_labels[i] == label1 and predicted_classes[i] == label1:
            correctly_classified_as_label1.append(probs[i])
        # Correctly classified as category 'label2'
        if test_labels[i] == label2 and predicted_classes[i] == label2:
            correctly_classified_as_label2.append(probs[i])
        # Incorrectly classified as category 'label1'
        if test_labels[i] == label2 and predicted_classes[i] == label1:
            incorrectly_classified_as_label1.append(probs[i])
        # Incorrectly classified as category 'label2'
        if test_labels[i] == label1 and predicted_classes[i] == label2:
            incorrectly_classified_as_label2.append(probs[i])
            
    # Plot probabilities and correctness for categories 0 and 1
    plt.figure()
    # Correctly classified as category 3
    plt.scatter(
        np.array(correctly_classified_as_label1)[:, 3],
        np.array(correctly_classified_as_label1)[:, 5],
        color="green",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label1}",
    )
    # Correctly classified as category 5
    plt.scatter(
        np.array(correctly_classified_as_label2)[:, 3],
        np.array(correctly_classified_as_label2)[:, 5],
        color="green",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label2}",
    )
    # Incorrectly classified as category 3
    plt.scatter(
        np.array(incorrectly_classified_as_label1)[:, 3],
        np.array(incorrectly_classified_as_label1)[:, 5],
        color="red",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label1}",
    )
    # Incorrectly classified as category 5
    plt.scatter(
        np.array(incorrectly_classified_as_label2)[:, 3],
        np.array(incorrectly_classified_as_label2)[:, 5],
        color="red",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label2}",
    )
    plt.plot([0, 1], [0, 1], color="black", label="Perfect classifier")
    plt.plot([1, 0], [1, 0], color="black")
    plt.title(title)
    plt.xlabel(f"Probability of category {label1}")
    plt.ylabel(f"Probability of category {label2}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()