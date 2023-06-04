import numpy as np
import subprocess
import os
import platform

# MCM_classifier helper imports
from .loaders import load_data, load_mcm
from .helpers import generate_bootstrap_samples, print_box


class MCM_Classifier:
    """
    The MCM-classifier

    Args:
        - n_categories (int): The number of categories in the dataset
        - n_variables (int): The number of variables in the dataset
    """

    def __init__(
        self,
        n_categories: int,
        n_variables: int,
        mcm_filename_format: str,
        data_filename_format: str,
    ) -> None:
        """
        The MCM-classifier.

        Args:
            - n_categories (int): The number of categories in the dataset
            - n_variables (int): The number of variables in the dataset
            - mcm_filename_format (str): The format of the MCM filenames
            - data_filename_format (str): The format of the data filenames
        """
        self.n_categories = n_categories
        self.n_variables = n_variables
        self.__mcm_filename_format = mcm_filename_format
        self.__data_filename_format = data_filename_format

        # Construct probability distributions and MCMs for each category
        self.__P, self.__MCM = ([], [])
        self.predicted_classes = None
        self.probs = None
        self.stats = None
        
    # ----- Public methods -----
    def init(self):
        """
        Initializes the classifier if the MCMs have already been selected.
        """
        self.__construct_P()

    def fit(self,
            data_path: str = "INPUT/data",
            greedy: bool = False,
            max_iter: int = 100000,
            max_no_improvement: int = 10000,
            n_samples: int = 0,
            ) -> None:
        """
        Fit the classifier using the data given in the data_path folder.
        It uses the MinCompSpin_SimulatedAnnealing algorithm to find the MCMs.

        Args:
            - data_path (str): Path to the data folder
            - greedy (bool): Whether to use the greedy algorithm after SA
            - max_iter (int): Maximum number of iterations for the SA algorithm
            - max_no_improvement (int): Maximum number of iterations without improvement for the SA algorithm
            - n_samples (int): The number of samples to be used from the data folder. If 0, all samples are used.
        """
        # if not self.__validate_input_data():
        #     raise ValueError("Input data folder file count does not match number of categories")
        # Loop over each file in the data folder
        folder = os.fsencode(data_path)
        sorted_folder = sorted(os.listdir(folder))
        
        for file in sorted_folder:
            filename = os.fsdecode(file)
            if filename.endswith(".dat"):
                # Remove the .dat extension
                if (n_samples == 0):
                    n_samples = len(load_data(data_path + "/" + filename))
                else:
                    # create new folder for bootstrap samples
                    bootstrap_name = filename[:-4] + "_bootstrap"
                    os.makedirs("INPUT/data/bootstrap/", exist_ok=True)
                    generate_bootstrap_samples(load_data("INPUT/data/" + filename), bootstrap_name, n_samples)
                    filename = "bootstrap/" + bootstrap_name + ".dat"

    
                filename = filename[:-4]
                file = "mcm_classifier/input/data/" + filename
                saa_args = self.__construct_args(file, greedy, max_iter, max_no_improvement)
                # Run the MinCompSpin_SimulatedAnnealing algorithm
                print(f"Running MinCompSpin_SimulatedAnnealing on {filename}...")
                p = subprocess.run(saa_args, stdout=subprocess.PIPE)
                print("Done!")
            else:
                continue
            
        # Construct probability distributions and MCMs for each category
        self.__construct_P()
   
    def classify(self, state: np.ndarray = np.array([])) -> tuple:
        """
        Classify a single state using the MCM-based classifier.
        
        Args:
            state (np.ndarray): The state to be classified
        """
        # ----- Calculate probability of sample belonging to each category -----
        probs = self.__get_probs(state)
        predicted_class = np.argmax(probs)
         
        return predicted_class, probs

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Evaluates the performance of the MCM-based classifier.

        Args:
            data (np.ndarray): The data to be classified
            labels (np.ndarray): The labels of the data
            
        Returns:
            tuple: The predicted classes (for each state) and the probabilities for each category (for each state)
        """
        print_box("Evaluating classifier...")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if len(self.__P) == 0 or len(self.__MCM) == 0:
            raise ValueError("Classifier not initialized yet. If you have already selected MCMs, try running the init method first. If not, try running the fit method first.")

        # ----- Calculate probability of sample belonging to each category -----
        print_box("1. Calculating state probabilities...")
        probs = np.array([self.__get_probs(state) for state in data])
        predicted_classes = np.argmax(probs, axis=1)

        # ----- Calculate accuracy -----
        print_box("2. Calculating accuracy...")

        correct_predictions = predicted_classes == labels
        accuracy = np.sum(correct_predictions) / len(labels)

        # ----- Save stats -----
        print_box("3. Saving stats...")

        self.predicted_classes = predicted_classes
        self.probs = probs
        self.stats = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "confusion_matrix": self.__get_confusion_matrix(labels),
        }

        print_box("Done!")

        return predicted_classes, probs

    def sample_MCM(self, n_samples: int):
        """
        Samples n_samples from the MCMs randomly
        
        Args:
            n_samples (int): The number of samples to be generated
        
        Returns:
            list: A list of the generated samples
        """
        samples = []

        for i in range(n_samples):
            category = np.random.randint(0, self.n_categories)
            samples.append(self.__sample_MCM(category))

        return samples

    def get_classification_report(self, labels: np.ndarray) -> dict:
        """
        Get the classification report for the classifier

        Args:
            labels (np.ndarray): The labels of the data

        Raises:
            ValueError: If the classifier has not been evaluated yet

        Returns:
            dict: The classification report
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        # Get the confusion matrix
        confusion_matrix = self.__get_confusion_matrix(labels)

        # Calculate the precision, recall and f1-score for each category
        precision = np.zeros(self.n_categories)
        recall = np.zeros(self.n_categories)
        f1_score = np.zeros(self.n_categories)

        for i in range(self.n_categories):
            # Calculate precision
            precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])

            # Calculate recall
            recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

            # Calculate f1-score
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        # Calculate the average precision, recall and f1-score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1_score = np.mean(f1_score)

        # Calculate the accuracy
        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

        # Construct the classification report
        classification_report = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1_score,
            "accuracy": accuracy,
        }

        return classification_report

    def save_classification_report(
        self, labels: np.ndarray, name: str = "classification_report", path: str = "OUTPUT"
    ) -> None:
        """
        Saves the classification report to a file
        
        Args:
            name (str): The name of the file
            labels (np.ndarray): The labels of the data
            path (str): The path to the folder where the file should be saved
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        # Get the classification report
        classification_report = self.get_classification_report(labels)
        self.__save_classification_report(name, classification_report, path)     

    # ----- Private methods -----
    def __construct_P(self) -> tuple:
        """
        Construct probability distributions for each category.
        This function should only be run once during initalization of the classifier.

        Code provided by https://github.com/ebokai

        Args:
            P (list): List of the probability distributions for each category
            MCM (list): List of the MCMs for each category
            n_categories (int): Number of categories in the dataset
            n_variables (int): Number of variables in the dataset
        """
        MCM = []
        P = []

        print_box("Constructing probability distributions...")

        # if not self.__validate_input_comms():
            # raise ValueError("Input data folder file count does not match number of categories. Did you run the fit method?.")

        # Construct probability distributions for each category
        for k in range(self.n_categories):
            # Add MCM to list
            try:
                mcm = load_mcm(f"INPUT/MCMs/{self.__mcm_filename_format.format(k)}")
                MCM.append(mcm)
            except:
                # Throw error if MCM file not found
                raise FileNotFoundError(f"Could not find MCM file for category {k}")

            # Load data
            try:
                data = load_data(f"INPUT/data/{self.__data_filename_format.format(k)}")
            except:
                # Throw error if data file not found
                raise FileNotFoundError(f"Could not find data file for category {k}")

            pk = []

            for icc in mcm:
                idx = [i for i in range(self.n_variables) if icc[i] == "1"]
                rank = len(idx)

                p_icc = np.zeros(2**rank)
                icc_data = data[:, idx]
                icc_strings = [
                    int("".join([str(s) for s in state]), 2) for state in icc_data
                ]

                u, c = np.unique(icc_strings, return_counts=True)

                p_icc[u] = c / np.sum(c)

                pk.append(p_icc)

            P.append(pk)

        self.__P = P
        self.__MCM = MCM

        return self.__P, self.__MCM

    def __construct_args(self,
                        filename: str,
                        greedy: bool,
                        max_iter: int,
                        max_no_improvement: int
                        ) -> tuple:
        """
        Generates the arguments for the MinCompSpin_SimulatedAnnealing algorithm

        Args:
            operating_system (str): _description_
            data_path (str): _description_
            greedy (bool): _description_
            max_iter (int): _description_
            max_no_improvement (int): _description_

        Returns:
            list: The list with all the arguments, to be used in the subprocess call
        """
        operating_system = platform.system()
        
        g = "-g" if greedy else ""
        
        sa_file = "../MinCompSpin_SimulatedAnnealing/bin/saa.exe" if operating_system == "Windows" else "../MinCompSpin_SimulatedAnnealing/bin/saa.out"
        saa_args = [sa_file,
                    str(self.n_variables),
                    '-i',
                    filename,
                    g,
                    '--max',
                    str(max_iter),
                    '--stop',
                    str(max_no_improvement)
        ]
        
        # Filter out empty strings
        saa_args = tuple(filter(None, saa_args))
        return saa_args
    
    def __validate_input_data(self, data_path: str = "INPUT/data",) -> bool:
        """
            Validates the input community folder. Checks if the number of files in the folder
            is equal to the number of categories.
        """
        folder = os.fsencode(data_path)
        sorted_folder = sorted(os.listdir(folder))
        
        n_matching_files = 0
        for file in sorted_folder:
            filename = os.fsdecode(file)
            print(filename)
            print(self.__data_filename_format.format(n_matching_files))
            if filename == self.__data_filename_format.format(n_matching_files):
                n_matching_files += 1

        if n_matching_files == self.n_categories: return True
        return False 
    
    def __validate_input_comms(self, comms_path: str = "INPUT/MCMs",) -> bool:
        """
            Validates the input community folder. Checks if the number of files in the folder
            is equal to the number of categories.
        """
        folder = os.fsencode(comms_path)
        sorted_folder = sorted(os.listdir(folder))
        
        n_matching_files = 0
        for i, file in enumerate(sorted_folder):
            filename = os.fsdecode(file)
            if filename == self.__mcm_filename_format.format(i):
                n_matching_files += 1

        if n_matching_files == self.n_categories: return True
        return False           
    
    def __sample_MCM(self, cat_index: int) -> np.ndarray:
        """
        Sample a state from some MCM.

        Args:
            cat_index (int): The category index from which to sample
        """
        # get a sample for each digit

        pk = self.__P[cat_index]  # probability distribution for each digit
        mcm = self.__MCM[cat_index]  # communities for each digit

        sampled_state = np.zeros(self.n_variables)

        for j, icc in enumerate(mcm):
            p_icc = pk[j]  # get the probability distribution restricted to specific ICC
            idx = [
                i for i in range(self.n_variables) if icc[i] == "1"
            ]  # count the number of variables in ICC
            rank = len(idx)
            sm = np.random.choice(np.arange(2**rank), 1, p=p_icc)[
                0
            ]  # sample "random" state of ICC
            ss = format(sm, f"0{rank}b")  # convert integer to binary string
            ss = np.array([int(s) for s in ss])  # convert binary string to [0,1] array
            sampled_state[idx] = ss  # fill ICC part of complete state

        return sampled_state

    def __prob_MCM(self, state: np.ndarray, cat_index: int) -> float:
        """
        Calculate the probability of a state given a single MCM.

        Args:
            P (np.ndarray): Probability distributions for one category
            MCM (np.ndarray): MCM for one category
            state (np.ndarray): The state to calculate the probability of
        """

        prob = 1
        MCM = self.__MCM[cat_index]
        P = self.__P[cat_index]

        # Loop through each ICC and calculate the probability of the state
        for j, icc in enumerate(MCM):
            # 1. Get the probability distribution restricted to specific ICC
            p_icc = P[j]
            idx = [i for i in range(self.n_variables) if icc[i] == "1"]
            # 2. Get the state of the variables in the ICC and convert to binary string
            ss = state[idx]
            sm = int("".join([str(s) for s in ss]), 2)

            # 3. Multiply the probability of the state by the probability of the ICC-state
            prob *= p_icc[sm]

        return prob

    def __get_probs(self, state: np.ndarray) -> list:
        """
        Get the probabilites for a single state for each category, in order
        
        Args:
            state (np.ndarray): The state to calculate the probability of
        """

        all_probs = []

        for i in range(self.n_categories):
            prob = self.__prob_MCM(state, i)
            all_probs.append(prob)

        return all_probs

    def __get_confusion_matrix(self, test_labels: np.ndarray):
        """
        Get the confusion matrix for the classifier
        
        Args:
            test_labels (np.ndarray): The labels of the test data
        
        Raises:
            ValueError: If the classifier has not been evaluated yet
        
        Returns:
            np.ndarray: The confusion matrix
        """
        if self.predicted_classes is None:
            raise ValueError("Classifier not evaluated yet")

        confusion_matrix = np.zeros((self.n_categories, self.n_categories))
        for i, label in enumerate(test_labels):
            confusion_matrix[label, self.predicted_classes[i]] += 1

        return confusion_matrix

    def __save_classification_report(self, name: str, classification_report: dict, path: str):
        """
        Saves the classification report to a file

        Args:
            name (str): The desired name of the file
            classification_report (dict): The classification report
            path (str): The path to the folder where the file should be saved
        """
        with open(f"{path}/{name}.txt", "w") as f:
            f.write("Classification report:\n")
            f.write(f"Accuracy: {classification_report['accuracy']}\n")
            f.write(f"Average precision: {classification_report['avg_precision']}\n")
            f.write(f"Average recall: {classification_report['avg_recall']}\n")
            f.write(f"Average f1-score: {classification_report['avg_f1_score']}\n")
            f.write("\n")
            f.write("Precision:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['precision'][i]}\n")
            f.write("\n")
            f.write("Recall:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['recall'][i]}\n")
            f.write("\n")
            f.write("F1-score:\n")
            for i in range(self.n_categories):
                f.write(f"{i}: {classification_report['f1_score'][i]}\n")

    def __str__(self) -> str:
        """
        String representation of the classifier
        """
        return f"MCM_Classifier(n_categories={self.n_categories}, n_variables={self.n_variables})"