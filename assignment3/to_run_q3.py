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

state_q3 = ... # in the form np.array([[...],[...],[...],[...],[...],[...]])

covar_q3 = ... # In the form np.array([[... , ... ,... , ... , ... , ...],[..., ... ,... , ..., ..., ...],... 

ID_q3 = ...

Cd = ...

Cr = ...

mass = ...

area = ...

####################################################################
####################################################################



print("##########################")
print("##########################")
print("QUESTION 3")
print("##########################")
print("##########################")

result_q3 = conj_ass_Q3(rso_catalog , ID, state_q3, covar_q3,ID_q3 , Cd_new = Cd , Cr_new = Cr , mass_new= mass , area_new= area )

