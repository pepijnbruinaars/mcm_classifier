import numpy as np
from .loaders import load_data, load_mcm

class MCM_Classifier:
    """
    Classify the data using the MCM-based classifier.
    
    Args:
        n_categories (int): The number of categories in the dataset
        n_variables (int): The number of variables in the dataset
    """
    def __init__(self, n_categories: int, n_variables: int) -> None:
        self.n_categories = n_categories
        self.n_variables = n_variables
        self.P, self.MCM = self.__construct_P()
        
    # ----- Public methods -----
    def classify(self) -> None:
        """
        Classify the data using the MCM-based classifier.

        Args:
            n_categories (int): The number of categories in the dataset
        """
        
        # ----- Sample from MCM -----
        print("Sampling from MCM...")
        sample = self.__sample_MCM()
        
        # ----- Calculate probability of sample belonging to each category -----
        print("Probability of sample 1 belonging to category 1:")
        print(self.__prob_MCM(self.P[0], self.MCM[0], sample[0].astype(int)))
        print("Probability of sample 1 belonging to category 2:")
        print(self.__prob_MCM(self.P[1], self.MCM[1], sample[0].astype(int)))
        print("Probability of sample 2 belonging to category 1:")
        print(self.__prob_MCM(self.P[0], self.MCM[0], sample[1].astype(int)))
        print("Probability of sample 2 belonging to category 2:")
        print(self.__prob_MCM(self.P[1], self.MCM[1], sample[1].astype(int)))
    
    def evaluate(self) -> None:
        """
        Evaluate the performance of the MCM-based classifier.
        """
        pass    

    # ----- Private methods -----
    def __construct_P(self) -> tuple:
        """
        Construct probability distributions for each category.
        
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
        for k in range(1, self.n_categories + 1):
            # Add MCM to list
            try:
                mcm = load_mcm(f'INPUT/MCMs/mcm-unlabeled-{k}.dat')
                MCM.append(mcm)
            except:
                # Throw error if MCM file not found
                raise FileNotFoundError(f'Could not find MCM file for category {k}')
            
            # Load data
            try:
                data = load_data(f'INPUT/data/train-images-unlabeled-{k}.dat')
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

    def __sample_MCM(self) -> None:
        """
        Sample a state from the MCM.

        Args:
            P (list): List of the probability distributions for each category
            MCM (list): List of the MCMs for each category
        """
        samples = []
        # get a sample for each digit
        for k in range(self.n_categories):

            pk = self.P[k] # probability distribution for each digit
            mcm = self.MCM[k] # communities for each digit
            
            sampled_state = np.zeros(121) # 121 = 11 x 11
            
            for j,icc in enumerate(mcm):
                
                p_icc = pk[j] # get the probability distribution restricted to specific ICC
                idx = [i for i in range(121) if icc[i] == '1'] # count the number of variables in ICC
                rank = len(idx)        
                sm = np.random.choice(np.arange(2**rank), 1, p = p_icc)[0] # sample "random" state of ICC
                ss = format(sm, f'0{rank}b') # convert integer to binary string
                ss = np.array([int(s) for s in ss]) # convert binary string to [0,1] array
                sampled_state[idx] = ss # fill ICC part of complete state
            samples.append(sampled_state)
            
        return samples
        
    def __prob_MCM(self, P: np.ndarray, MCM: np.ndarray, state: np.ndarray) -> float:
        """
        Calculate the probability of a state given a single MCM.

        Args:
            P (np.ndarray): Probability distributions for one category
            MCM (np.ndarray): MCM for one category
            state (np.ndarray): The state to calculate the probability of
        """
        
        prob = 1
        
        # Loop through each ICC and calculate the probability of the state
        for j,icc in enumerate(MCM):
            # Get the probability distribution restricted to specific ICC
            p_icc = P[j]
            idx = [i for i in range(121) if icc[i] == '1']
            # Get the state of the variables in the ICC and convert to binary string
            ss = state[idx]
            sm = int(''.join([str(s) for s in ss]),2)
            # Multiply the probability of the state by the probability of the ICC-state
            prob *= p_icc[sm]
                        
        return prob