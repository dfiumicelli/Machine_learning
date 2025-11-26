import numpy as np


def gaussian_kernel(x1, x2, sigma):
    """
    Returns the similarity between x1 and x2 using a Gaussian kernel.

    Parameters
    ----------
    x1 : ndarray
        Sample 1.
    x2 : ndarray
        Sample 2.
    sigma : float
        Bandwidth for Gaussian kernel.

    Returns
    -------
    float
        The similarity between x1 and x2 with bandwidth sigma.
    """
    # Ci assicuriamo che x1 e x2 siano array numpy
    x1 = np.array(x1)
    x2 = np.array(x2)

    # Calcoliamo la differenza euclidea tra i due vettori
    diff = x1 - x2
    squared_distance = np.dot(diff, diff)  #norma al quadrato

    # Formula del kernel gaussiano
    sim = np.exp(-squared_distance / (2 * sigma ** 2))

    return sim
