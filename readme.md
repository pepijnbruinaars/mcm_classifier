# MCM-Classifier
This is a MCM-based naive Bayes' classifier. It is a very simple classifier that uses fitted Minimally Complex Models to classify data. More information on MCMs and the way they're fitted can be found [here](https://github.com/clelidm/MinCompSpin), [here](https://github.com/clelidm/MinCompSpin_Greedy) and [here](https://github.com/ebokai/MinCompSpin_SimulatedAnnealing).

Sample data from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is given inside the `INPUT` folder. As MCMs only supports binary datasets with 128 or less features, the data was pre-processed by cutting of the edges coarse graining the images into 11x11 binary matrices, where 1 means a given pixel is lit, and 0 means it isn't. The [simulated annealing](https://github.com/ebokai/MinCompSpin_SimulatedAnnealing) method has been applied to this dataset and using the resulting MCMs results in an accuracy of approximately 90%.

## Usage
In order to use this classifier, one should have:
- A <ins>uniform</ins> dataset, with each row being a sample and each column being a feature, preferabbly separated per label and numbered accordingly.
- A set of labels, one for each sample in the dataset.
- The communities of a fitted MCM for each label, assumed to be in the `INPUT/MCMs` folder. In order to select your MCMs, I'd recommend using the [simulated annealing](https://github.com/ebokai/MinCompSpin_SimulatedAnnealing) selection method.

In `main.py`, one has to set the number of labels present in the dataset, as well as the number of features. Finally, you should name all files according to the format defined above (i.e. replace `"train-images-unlabeled-{}_comms.dat"` with anything else so it matches your filenames, and place the `{}` in the place where your number is in the filename.)

## Requirements and dependencies
The code requires you to have Python installed.

This classifier also depends on the following packages:
- NumPy: `pip install numpy`

## Todo-list
- Fitting MCMs for data in the program itself.
