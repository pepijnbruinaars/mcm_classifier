import numpy as np

# Provided by (https://github.com/ebokai)
def construct_P():
    P = [] # probability distributions for each mcm
    MCM = []

    for k in range(10):

        mcm = np.loadtxt(f'../data/train-images-no-label-{k}.txt_comms.dat', dtype=str)
        
        MCM.append(mcm)
        
        data = np.loadtxt(f'../data/train-images-no-label-{k}.txt', dtype=str)
        data = np.array([[int(s) for s in state] for state in data])
        
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