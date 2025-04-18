# Lambert targeting for an
# Earth to Mars transfer

# Import required modules
import numpy as np
import tudatpy
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel import constants

# Given constants
sun_gm = 132.71244e18 # [m^3/s^2]

# Given initial values
initial_epoch = 0.0 # [s]
final_epoch = 600.0 * constants.JULIAN_DAY # [s]
departure_pos_initial_epoch = [81.319e+9, -116.36e+9, -50.43e+9] # [m]
target_pos_final_epoch = [24.935e+9, -193.9e+9, -89.63e+9] # [m]

# Solve Lambert's problem
lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
	departure_position = departure_pos_initial_epoch,
	arrival_position = target_pos_final_epoch,
	time_of_flight = final_epoch - initial_epoch,
	gravitational_parameter = sun_gm)

# Extract results
v1, v2 = lambert_targeter.get_velocity_vectors()

# Print the results
np.set_printoptions(formatter={'float': '{: 9.3e}'.format})
print('|V1|{tab} = {v1:6.3f}{tab} [km/s] {cr}|V2|{tab} = {v2:6.3f}{tab} [km/s]'.format(v1=np.linalg.norm(v1)/1e3, v2=np.linalg.norm(v2)/1e3, cr='\n', tab='\t'))
print(' V1{tab} = {v1}{tab} [km/s] {cr} V2{tab} = {v2}{tab} [km/s]'.format(v1 = np.array2string(v1/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), v2 = np.array2string(v2/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), cr='\n', tab='\t'))
