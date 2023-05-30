import numpy as np
from .loaders import load_data, load_mcm

def construct_P(P: list, MCM: list, n_categories: int) -> None:
    """
    Construct probability distributions for each category.
    
    Code provided by https://github.com/ebokai
    
    Args:
        P (list): List of the probability distributions for each category
        MCM (list): List of the MCMs for each category
        n_categories (int): Number of categories in the dataset
    """
    # Construct probability distributions for each category
    for k in range(n_categories):

        mcm = load_mcm(f'../data/train-images-no-label-{k}.txt_comms.dat')
        
        MCM.append(mcm)
        
        data = load_data(f'../data/train-images-no-label-{k}.txt')
        
        pk = []

        for icc in mcm:
            
            idx = [i for i in range(121) if icc[i] == '1']
            rank = len(idx)
            
            p_icc = np.zeros(2**rank)
            icc_data = data[:,idx]
            icc_strings = [int(''.join([str(s) for s in state]),2) for state in icc_data]
            
            u, c = np.unique(icc_strings, return_counts = True)

            p_icc[u] = c / np.sum(c)
            
            pk.append(p_icc)
        
        P.append(pk)

# Provided by (https://github.com/ebokai)
def sample_MCM(P, MCM):
    """
    Sample a state from the MCM.

    Args:
        P (_type_): _description_
        MCM (_type_): _description_
    """
    # get a sample for each digit
    for k in range(10):


        pk = P[k] # probability distribution for each digit
        mcm = MCM[k] # communities for each digit
        
        sampled_state = np.zeros(121) # 121 = 11 x 11
        
        for j,icc in enumerate(mcm):
            
            p_icc = pk[j] # get the probability distribution restricted to specific ICC
            idx = [i for i in range(121) if icc[i] == '1'] # count the number of variables in ICC
            rank = len(idx)        
            sm = np.random.choice(np.arange(2**rank), 1, p = p_icc)[0] # sample "random" state of ICC
            ss = format(sm, f'0{rank}b') # convert integer to binary string
            ss = np.array([int(s) for s in ss]) # convert binary string to [0,1] array
            sampled_state[idx] = ss # fill ICC part of complete state
            
            