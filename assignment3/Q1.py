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
    step = 100,
    max_step = 3600,
    min_step = 1e-3,
    rtol = 1e-12,
    atol = 1e-12
)
T = np.zeros((100, 100)) 
rho = np.zeros((100, 100))

t_range = [tdb_epoch , tdb_epoch + 2*constants.JULIAN_DAY]


result = conjunction_assessment(rso_catalog , 30e3, ID)

# Save the result list of dictionaries to a file
output_file = 'conjunction_assessment_results.pkl'

with open(output_file, 'wb') as file:
    pickle.dump(result, file)

print(f"Results saved to {output_file}")


# Filter out dictionaries with empty values
filtered_result = [entry for entry in result if entry['Pc'] or entry['distance_at_tca']]

# Save the filtered result list of dictionaries to a file
filtered_output_file = 'filtered_conjunction_assessment_results.pkl'

with open(filtered_output_file, 'wb') as file:
    pickle.dump(filtered_result, file)

print(f"Filtered results saved to {filtered_output_file}")
print(filtered_result)