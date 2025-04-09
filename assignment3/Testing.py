import pickle
import numpy as np
from tudatpy import util
from tudatpy import constants
import ConjunctionUtilities
import TudatPropagator
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
from scipy.stats import norm, chi2
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
# Path to the pickle file
import matplotlib.pyplot as plt

def plot_sigma_ellipsoid(ax, cov_matrix, center, label, color, sigma=2):
    """
    Plot the ellipsoid representing a sigma region.
    
    Parameters:
        ax (Axes3D): The 3D plot axes.
        cov_matrix (ndarray): The covariance matrix.
        center (ndarray): The center of the ellipsoid.
        label (str): The label for the ellipsoid.
        color (str): The color of the ellipsoid.
        sigma (int): The sigma level (e.g., 2 for 2-sigma, 3 for 3-sigma).
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Generate points for the ellipsoid (unit sphere)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Scale the points using the square root of eigenvalues and the desired sigma level
    x = sigma * np.sqrt(eigenvalues[0]) * x
    y = sigma * np.sqrt(eigenvalues[1]) * y
    z = sigma * np.sqrt(eigenvalues[2]) * z
    
    # Rotate the points by eigenvectors
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                eigenvectors @ np.array([x[i, j], y[i, j], z[i, j]]).T
            )
    
    # Translate to the center
    x += center[0]
    y += center[1]
    z += center[2]
    
    # Plot the ellipsoid using a wireframe
    ax.plot_wireframe(x, y, z, color=color, alpha=0.5, label=f'{sigma}-sigma region: {label}')
    ax.legend()
