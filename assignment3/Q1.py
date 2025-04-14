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

print(rso_catalog)
# F1_ids = perigee_apogee_filter(rso_catalog , 40697)

# filtered_rso_catalog_Apogee = {key: rso_catalog[key] for key in F1_ids}


# population_analysis(rso_catalog , filtered_rso_catalog_Apogee  , filtered_rso_catalog_Apogee, ID)


# LEO_id, MEO_id, GEO_id, HEO_id = full_catalog_analysis(rso_catalog)
# print("LEO:", LEO_id)
# print("MEO:", MEO_id)
# print("GEO:", GEO_id)
# print("HEO:", HEO_id)


ids = rso_catalog.keys()
state_ref = rso_catalog[ID]['state']
tdb_epoch = rso_catalog[ID]['epoch_tdb']
Cd_1 = rso_catalog[ID]['Cd']
Cr_1 = rso_catalog[ID]['Cr']
area_1 = rso_catalog[ID]['area']
mass_1 = rso_catalog[ID]['mass']

# Define additional parameters for the propagation

sph_deg = 8

sph_ord = 8

central_bodies = ['Earth']

bodies_to_create = ['Earth', 'Sun', 'Moon']
state_params_1 = dict(
    central_bodies = central_bodies , 
    bodies_to_create = bodies_to_create , 
    mass = mass_1 , area = area_1 , 
    Cd = Cd_1 , Cr = Cr_1 , 
    sph_deg = sph_deg , 
    sph_ord = sph_ord
    )

# Define integrator parameters = dict(step , max_step, min_step,rtol, atol, tudat_integrator)
int_params = dict(
    tudat_integrator = 'rkf78',
    step = 10,
    max_step = 3600,
    min_step = 1e-3,
    rtol = 1e-12,
    atol = 1e-12
)
T = np.zeros((100, 100)) 
rho = np.zeros((100, 100))

t_range = [tdb_epoch , tdb_epoch + 2*constants.JULIAN_DAY]




result = conjunction_assessment(rso_catalog, ID)
# # # # # # Save the result to a file for later access
# result_file = 'conjunction_assessment_result.pkl'
# with open(result_file, 'wb') as file:
#     pickle.dump(result, file)

# print(f"Conjunction assessment results saved to {result_file}")

# # # Load the result file for the next function
# with open(result_file, 'rb') as file:
#     loaded_result = pickle.load(file)

# processing_results_gaussian(loaded_result, rso_catalog, ID)

plot_3D_orbits(rso_catalog , ID , result)




##########################################

# LETs define a framework for all the other c's: 
# FOR Q3 and Q4 I can just modify the full catalog and run the entire thing again to see if there is 
# a new possible collision.

# THe only problem is apparently Q2, as the object 91159 will perform a maneuver in an epoch later than the initial one provided. 
# The latter has already been analysed, and it will (without any maneuver), closely encounter the reference ID. 

# # Supposedly you get the delta-v at a certain epoch, with supposedly a modified state and covariance? 

# state_91159 = ...

# cov_91159 = ...

# epoch_tdb_91159 = ...

# # Those are the only thing that actually changes. In my opinion, the smart way of doing this is creating a catalog with just the object, and analyzing 
# # it from the start of the maneuver to the end of the 2 days time span. The time from the initial epoch to the maneuver itself will not be analyzed if 
# # the objects has not encountered the main object in that time span (which is the case, see report)

# result = conj_ass_Q2(rso_catalog , ID , 91159 , epoch_tdb_91159 , state_91159 , cov_91159) # To run (?)


## For the rest, I dont really know what it should be the output but in general
ID_q3 = 91662
# Q3 
Cd = 3
Cr = 1.3
mass = mass_1
area= 1

print("##########################")
print("##########################")
print("QUESTION 3")
print("##########################")
print("##########################")

result_q3 = conj_ass_Q3(rso_catalog , ID, ID_q3 , Cd_new = Cd , Cr_new = Cr , mass_new= mass , area_new= area )


# #Q4 
# ID_q4 = 99005
# state_mean_new = ...
# cov_new = ...
# rso_catalog[ID_q4]['state'] = state_mean_new
# rso_catalog[ID_q4]['cov'] = cov_new

# result_q4 = conjunction_assessment(rso_catalog , ID)