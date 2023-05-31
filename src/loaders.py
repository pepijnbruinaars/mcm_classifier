import numpy as np

def load_data(path: str) -> np.ndarray:
    """
    Load data from a given path.
    
    Args:
        path (str): Pathname of the file containing the data
        
    Returns:
        np.ndarray: The data loaded from the given path
    """
    data = np.loadtxt(path, dtype=str)
    data = np.array([[int(s) for s in state] for state in data])
    
    return data

def load_labels(path: str) -> np.ndarray:
    """
    Loads labels from a given path.

    Args:
        path (str): Pathname of the file containing the labels

    Returns:
        np.ndarray: 1-D array containing the labels of test data
    """
    labels = np.loadtxt(path, dtype=str)
    
    return labels

def load_mcm(path: str) -> np.ndarray:
    """
    Loads an MCM from a given path.

    Args:
        path (str): Pathname of the file containing the MCM

    Returns:
        np.ndarray: The MCM loaded from the given path
    """
    mcm = np.loadtxt(path, dtype=str)
    
    return mcm