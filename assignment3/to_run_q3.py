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
from Filters import *
# Path to the pickle file
file_path = 'assignment3/data/group5/estimated_rso_catalog.pkl'

# Load the pickle file
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except pickle.UnpicklingError:
    print("Error unpickling the file.")

rso_catalog =  data[0]
ID = 40697  # ID of the to-defend-object

Cd_1 = rso_catalog[ID]['Cd']
Cr_1 = rso_catalog[ID]['Cr']
area_1 = rso_catalog[ID]['area']
mass_1 = rso_catalog[ID]['mass']

###################################################################
###################################################################

# TO MODIFY FOR Q3 

state_q3 = np.array([[ 6.39112337e+06],
       [-1.46880903e+06],
       [-2.87145905e+06],
       [ 3.16955426e+03],
       [ 5.07698730e+02],
       [ 6.73782526e+03]]) # in the form np.array([[...],[...],[...],[...],[...],[...]])

covar_q3 = np.array([[ 5.73462722e-01,  8.01365105e-02,  1.04698784e-01,
        -2.77580190e-04, -2.43913472e-06, -3.12695278e-04],
       [ 8.01365105e-02,  9.52283301e-01, -3.66865291e-02,
         4.31060969e-05,  5.17283495e-06,  7.96481163e-05],
       [ 1.04698784e-01, -3.66865291e-02,  8.40405912e-01,
        -1.19701140e-05,  3.09983642e-05,  1.91972854e-04],
       [-2.77580190e-04,  4.31060969e-05, -1.19701140e-05,
         7.60535615e-07,  1.20535022e-08, -1.47531722e-07],
       [-2.43913472e-06,  5.17283495e-06,  3.09983642e-05,
         1.20535022e-08,  9.65656949e-07, -3.66705369e-08],
       [-3.12695278e-04,  7.96481163e-05,  1.91972854e-04,
        -1.47531722e-07, -3.66705369e-08,  5.73445208e-07]]) # In the form np.array([[... , ... ,... , ... , ... , ...],[..., ... ,... , ..., ..., ...],... 

ID_q3 = 91662

Cd = 2.2

Cr = 1.3

mass = 100

area = 12.3726661

####################################################################
####################################################################



print("##########################")
print("##########################")
print("QUESTION 3")
print("##########################")
print("##########################")

result_q3 = conj_ass_Q3(rso_catalog , ID, state_q3, covar_q3,ID_q3 , Cd_new = Cd , Cr_new = Cr , mass_new= mass , area_new= area )

