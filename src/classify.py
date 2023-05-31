import numpy as np
from .loaders import load_data, load_mcm

class MCM_Classifier:
    """
    The MCM-classifier
    
    Args:
        - n_categories (int): The number of categories in the dataset
        - n_variables (int): The number of variables in the dataset
    """
    def __init__(self, n_categories: int, n_variables: int, 
                 mcm_filename_format: str, data_filename_format: str) -> None:
        """
        Init function, constructs the classifier.

        Args:
            - n_categories (int): The number of categories in the dataset
            - n_variables (int): The number of variables in the dataset
        """
        self.n_categories = n_categories
        self.n_variables = n_variables
        self.__mcm_filename_format = mcm_filename_format
        self.__data_filename_format = data_filename_format
        
        # Construct probability distributions and MCMs for each category
        self.P, self.MCM = self.__construct_P()
        
    # ----- Public methods -----
    def classify(self, state: np.ndarray = np.array([])) -> None:
        """
        Classify a single state using the MCM-based classifier.
        """
        # ----- Calculate probability of sample belonging to each category -----
        probs = self.__get_probs(state)
        predicted_class = np.argmax(probs)
        print(f"Predicted class: {predicted_class} with probability {probs[predicted_class]}")
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        Evaluates the performance of the MCM-based classifier.
        
        Args:
            data (np.ndarray): The data to be classified
            labels (np.ndarray): The labels of the data
        """
        # ----- Calculate probability of sample belonging to each category -----
        probs = np.array([self.__get_probs(state) for state in data])
        predicted_classes = np.argmax(probs, axis=1)
        
        # ----- Calculate accuracy -----
        correct_predictions = (predicted_classes == labels)
        accuracy = np.sum(correct_predictions) / len(labels)
        
        print(f"Accuracy: {accuracy}")
        
        return predicted_classes, probs, accuracy
    
    def sample_MCM(self, n_samples: int):
        """
        Samples n_samples from the MCMs randomly
        """
        samples = []
        
        for i in range(n_samples):
            category = np.random.randint(0, self.n_categories)
            samples.append(self.__sample_MCM(category))
            
        return samples
            
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
        # Construct probability distributions for each category
        for k in range(self.n_categories):
            # Add MCM to list
            try:
                mcm = load_mcm(f'INPUT/MCMs/{self.__mcm_filename_format.format(k)}')
                MCM.append(mcm)
                print(f"Loaded MCM for category {k}...")
            except:
                # Throw error if MCM file not found
                raise FileNotFoundError(f'Could not find MCM file for category {k}')
            
            # Load data
            try:
                data = load_data(f'INPUT/data/{self.__data_filename_format.format(k)}')
                print(f"Loaded data for category {k}...")
            except:
                # Throw error if data file not found
                raise FileNotFoundError(f'Could not find data file for category {k}')
            
            pk = []
    
            for icc in mcm:
                
                idx = [i for i in range(self.n_variables) if icc[i] == '1']
                rank = len(idx)
                
                p_icc = np.zeros(2**rank)
                icc_data = data[:,idx]
                icc_strings = [int(''.join([str(s) for s in state]),2) for state in icc_data]
                
                u, c = np.unique(icc_strings, return_counts = True)

                p_icc[u] = c / np.sum(c)
                
                pk.append(p_icc)
            
            P.append(pk)
            
        self.P = P
        self.MCM = MCM
            
        return self.P, self.MCM

    def __sample_MCM(self, cat_index: int) -> None:
        """
        Sample a state from some MCM.

        Args:
            cat_index (int): The category index from which to sample
        """
        # get a sample for each digit

        pk = self.P[cat_index] # probability distribution for each digit
        mcm = self.MCM[cat_index] # communities for each digit
        
        sampled_state = np.zeros(self.n_variables)
        
        for j,icc in enumerate(mcm):
            
            p_icc = pk[j] # get the probability distribution restricted to specific ICC
            idx = [i for i in range(self.n_variables) if icc[i] == '1'] # count the number of variables in ICC
            rank = len(idx)        
            sm = np.random.choice(np.arange(2**rank), 1, p = p_icc)[0] # sample "random" state of ICC
            ss = format(sm, f'0{rank}b') # convert integer to binary string
            ss = np.array([int(s) for s in ss]) # convert binary string to [0,1] array
            sampled_state[idx] = ss # fill ICC part of complete state
        
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
        MCM = self.MCM[cat_index]
        P = self.P[cat_index]
        
        # Loop through each ICC and calculate the probability of the state
        for j,icc in enumerate(MCM):
            # 1. Get the probability distribution restricted to specific ICC
            p_icc = P[j]
            idx = [i for i in range(self.n_variables) if icc[i] == '1']
            # 2. Get the state of the variables in the ICC and convert to binary string
            ss = state[idx]
            sm = int(''.join([str(s) for s in ss]), 2)
            
            # 3. Multiply the probability of the state by the probability of the ICC-state
            prob *= p_icc[sm]
                        
        return prob

    def __get_probs(self, state):
        """
        Get the probabilites for a single state for each category, in order
        """
        
        all_probs = []

        for i in range(self.n_categories):
            prob = self.__prob_MCM(state, i)
            all_probs.append(prob)
            
        return all_probs