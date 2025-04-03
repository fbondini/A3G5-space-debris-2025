###############################################################################
# This file contains a series of functions used to filter the population of RSO
###############################################################################

import numpy as np
import math
from datetime import datetime
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.special import erfcinv
from scipy.optimize import fsolve
import pickle
import time
import os
import matplotlib.pyplot as plt
from tudatpy import astro
from tudatpy import constants
import ConjunctionUtilities

import TudatPropagator 

# Constants
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
Re = 6378137  # Earth's radius (m)
omega_e = 7.2921159e-5  # Earth's rotation rate (rad/s)
from scipy.constants import g as g0, R as R_universal
R_specific = R_universal / 0.0289644  # Specific gas constant for air (J/(kg·K))
def atmospheric_density(a ,e ,v):
    """
    Computes the atmospheric density based on altitude using an exponential model
    with tabulated values from the U.S. Standard Atmosphere, 1976.
    """
    rad = r(a, e ,v)
    h = height_from_radius(rad)
    # Tabulated altitudes (km) and corresponding densities (kg/m^3)
    altitudes = np.array([0, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    densities = np.array([1.2250, 0.0190, 1.4275e-4, 5.297e-7, 6.967e-9, 2.5407e-10, 6.0706e-11, 2.9710e-12, 3.6662e-13, 2.8028e-12, 7.2070e-13, 2.8028e-13, 1.1367e-13, 6.5228e-14, 5.7114e-14, 3.8993e-14, 2.5970e-14, 1.1336e-14, 8.9012e-15, 5.8295e-15, 3.6698e-15, 3.5611e-15])
    scale_heights = np.array([7.249, 6.349, 5.877, 5.799, 5.382, 5.799, 7.249, 6.349, 5.877, 5.382, 5.799, 7.249, 6.349, 5.877, 5.382, 5.799, 7.249, 6.349, 5.877, 5.382, 5.799, 7.249])

    # Convert altitude from meters to kilometers
    h = h / 1000.0
    # Check altitude range and assign constant density
    if h < 0:
        return densities[0]  # Below the first layer
    elif h > 1000:
        return 0.0  # Above 1000 km, density is zero

    # Find the layer index for the given altitude
    layer = np.searchsorted(altitudes, h) - 1
    if layer < 0:
        return densities[0]  # Below the first layer
    elif layer >= len(densities) - 1:
        return densities[-1]  # Above the last defined layer

    # Use constant density for the layer
    print(f"density {densities[layer]}")
    return densities[layer]



def height_from_radius(r):
    """
    Calculates the height above the Earth's surface given the radial distance.
    Takes into account the ellipsoidal shape of the Earth.
    """
    Re_equator = 6378000  # Earth's equatorial radius (m)
   
    h = r - Re
    print(f"height {h}")
    return h


def velocity(a, e, v):
    """
    Computes the satellite velocity relative to the atmosphere.
    """
    p = a * (1 - e**2)
    V = np.sqrt(mu * (1 + e**2 + 2 * e * np.cos(v)) / p)
    return V


def r(a, e, v):
    """
    Calculate the orbital radius for given semi-major axis, eccentricity, and true anomaly.
    """
    return a * (1 - e**2) / (1 + e * np.cos(v))

def da_dt(a, e, i, B):
    """
    Computes the rate of change of the semi-major axis (da/dt) due to atmospheric drag.
    """

    integrand = lambda v: atmospheric_density(a , e, v) * velocity(a, e, v) * (
        1 + e**2 + 2 * e * np.cos(v) - omega_e * np.cos(i) * np.sqrt(
        (a**3 * (1 - e**2)**3) / mu) )* (r(a, e, v)**2 / (a * (1 - e**2)**(3/2)))
    result, _ = quad(integrand, 0, 2 * np.pi)
    return -B / (2 * np.pi) * result


def de_dt(a, e, i, B):
    """
    Computes the rate of change of the eccentricity (de/dt) due to atmospheric drag.
    """
    integrand = lambda v: atmospheric_density(a , e, v) * velocity(a, e, v) * (
        e + np.cos(v) - (r(a, e, v)**2 * omega_e * np.cos(i)) / (2 * np.sqrt(mu * a * (1 - e**2))) * (2 * (e + np.cos(v)) - e * np.sin(v)**2)) *(
        (r(a, e, v) / a)**2 * (1 - e**2)**(-1/2))
    result, _ = quad(integrand, 0, 2 * np.pi)
    return -B / (2 * np.pi) * result



def J2_J3_perturbations(a, e, i , omega):
    """
    Computes the J2 and J3 perturbation effects on the rates of change of the orbital elements.
    """
    J2 = 1.08263e-3
    J3 = -2.5327e-6
    p = a * (1 - e**2)
    n = np.sqrt(a**3 / mu)

    # Perturbation terms due to J2
    dOmega_J2 = -3/2 *n* J2 * (Re / p)**2 * np.cos(i) # Ok

    domega_J2 = 3/4 * n* J2 * (Re / p)**2 * (4 - 5 * np.sin(i)**2) # ok

    # Perturbation terms due to J3
    dOmega_J3 = 0 # ok
    domega_J3 = 3/8 * n* J3 * (Re / p)**3 *(  (4 - 5 * np.sin(i)**2)* ((np.sin(i) - e**2*np.cos(i)**2)/(e*np.sin(i))) + 2*np.sin(i)*(13-15*np.sin(i)**2)*e)*np.sin(omega)  #Ok
    de_J3 = -3/8*n*J3*(Re/p)**3*np.sin(i)*(4-5*np.sin(i)**2)*(1-e**2)*np.cos(omega)

    return dOmega_J2 + dOmega_J3, domega_J2 + domega_J3 , de_J3


def orbital_elements_rate(a, e, i, omega, B):
    """
    Calculates the rate of change of all orbital elements.
    """
    da = da_dt(a, e, i, B)
    de = de_dt(a, e, i, B)
    di = 0  # Inclination rate of change is zero for the simplified model
    dOmega_J2_J3, domega_J2_J3 , de_J3 = J2_J3_perturbations(a, e, i, omega)
    dOmega = dOmega_J2_J3
    domega =  domega_J2_J3
    de = de + de_J3

    return da, de, di, dOmega, domega










def perigee_apogee_filter(rso_catalog, D, ID_ref):
    print("Start of the perigee-apogee screening")
    filtered_ids = []
    IDs = rso_catalog.keys()
    print(f"Starting population dimension: {len(IDs)}")
    # Cycle within the different objects
    state_ref = rso_catalog[ID_ref]['state']
    Cd_ref = rso_catalog[ID_ref]['Cd']
    A_ref = rso_catalog[ID_ref]['area']
    m_ref = rso_catalog[ID_ref]['mass']
    B_ref = Cd_ref*(A_ref/m_ref)

    kepler = astro.element_conversion.cartesian_to_keplerian(state_ref, 3.986004415e14)

    da, de, di, dOmega, domega = orbital_elements_rate(kepler[0], kepler[1], kepler[2], kepler[4],B_ref)

    hyp_da = da*2*constants.JULIAN_DAY

    print(f"Semimajor axis variation in m {hyp_da}")

    ae_ref = [kepler[0], kepler[1]]

    Rp_ref = ae_ref[0] * (1 - ae_ref[1])

    Ra_ref = ae_ref[0] * (1 + ae_ref[1])
    print(Ra_ref/1000 -6371)
    print(Rp_ref/1000 -6371)

    states = np.zeros((len(IDs), 6))
    ae = np.zeros((len(IDs), 2))
    Rp= np.zeros(len(IDs))
    Ra = np.zeros_like(Rp)

    for i in range(len(IDs)):
        if list(IDs)[i] != ID_ref:
            states[i,:] = np.array(rso_catalog[list(IDs)[i]]['state']).flatten()  # Get the state of ALL the other objects
            kepler = astro.element_conversion.cartesian_to_keplerian(states[i,:], 3.986004415e14)
            ae[i, : ] = [kepler[0] , kepler[1]]
            Rp[i] = ae[i,0]*(1 - ae[i,1])
            Ra[i] = ae[i,0]*(1 + ae[i,1])

            # Determine which of the Rp is higher
            if Rp_ref > Rp[i]:
                q = Rp_ref
            else:
                q = Rp[i]

            if Ra_ref < Ra[i]:
                Q = Ra_ref
            else:
                Q = Ra[i]
            
            print(q-Q)
            ## Check if q - Q > D
            if np.abs(q-Q) > D:
                print(f"Object {list(IDs)[i]} is filtered")
                filtered_ids.append(list(IDs)[i])

    perc_of_filter = (len(filtered_ids))/(len(IDs)) * 100
    non_filtered_ids = [ID for ID in IDs if ID not in filtered_ids]
    print(f"Filtering completed for the datased. About {perc_of_filter}% of objects have been discarted")
    print(f"Number of remaining objects: {len(non_filtered_ids)}")
    print(f"Objects survived: {non_filtered_ids}")
    indices_non_filtered = [list(IDs).index(ID) for ID in non_filtered_ids]
    print(f"Indices of non-filtered objects: {indices_non_filtered}")
    for index in indices_non_filtered:
        print(f"Object {list(IDs)[index]}: Ra = {Ra[index]/1000 - 6371}, Rp = {Rp[index]/1000 - 6371}")
    return non_filtered_ids

def geometric_filter():
    return

def time_filter(rso_catalog , D , ID_ref):
    print("Start of the time screening")
    filtered_ids = []
    cooplanar_ids = []
    IDs = rso_catalog.keys()
    print(f"Starting population dimension: {len(IDs)}")
    state_ref = rso_catalog[ID_ref]['state']

    kepler_ref = astro.element_conversion.cartesian_to_keplerian(state_ref, 3.986004415e14)
    a_p = kepler_ref[0]
    e_p = kepler_ref[1]
    I_p = kepler_ref[2]
    w_p = kepler_ref[3]
    Omega_p = kepler_ref[4]
    theta_p = kepler_ref[5]
    P_p = 2*np.pi*np.sqrt(a_p**3/ 3.986004415e14)
    rev_p = (2 * constants.JULIAN_DAY) / P_p   # The number of revolutions will define the number of time intervals I can generate wihin the time period
    rev_p_integer = int(rev_p)  
    rev_p_decimal = rev_p - rev_p_integer 
    print(f"The to-defend object has a period of {P_p} seconds.") 
    print(f"Given the interval of 48 hours, this is {rev_p_integer} full revolutions and {rev_p_decimal}!")

    for i in range(len(IDs)):
        if list(IDs)[i] != ID_ref:
            ID_2 = list(IDs)[i]
            print(f"Analyzing object {ID_2}...")
            state = rso_catalog[ID_2]['state']
            kepler = astro.element_conversion.cartesian_to_keplerian(state, 3.986004415e14)
            print(f"COE retreived")
            a_s = kepler[0]
            e_s = kepler[1]
            I_s = kepler[2]
            w_s = kepler[3]
            Omega_s = kepler[4]
            theta_s = kepler[5]
            P_s = 2*np.pi*np.sqrt(a_s**3/ 3.986004415e14)

            rev_s = (2 * constants.JULIAN_DAY) / P_s
            rev_s_integer = int(rev_s)
            rev_s_decimal = rev_s - rev_s_integer

            I_r = compute_I_r(I_p , I_s , Omega_p , Omega_s)
            
            print(f"Relative inclination of {np.rad2deg(I_r)} degree")

            delta_p , delta_s = compute_delta(I_p , I_s , Omega_p , Omega_s)

            ax_p , ay_p = compute_a_vectors(e_p , w_p , delta_p)

            ax_s , ay_s = compute_a_vectors(e_s , w_s , delta_s)

            ur_p1 , ur_p2 , ur_p3 , ur_p4 = compute_u_r(a_p , e_p, I_r , ax_p , ay_p , D)

            if ur_p1 == -30:
                cooplanar_ids.append(ID_2)

            int_p_1 = [ur_p1 , ur_p2]
            int_p_2 = [ur_p3 , ur_p4]

            ur_s1 , ur_s2 , ur_s3 , ur_s4 = compute_u_r(a_s , e_s, I_r , ax_s , ay_s , D)

            if ur_s1 == -30:
                cooplanar_ids.append(ID_2)

            int_s_1 = [ur_s1 , ur_s2]
            int_s_2 = [ur_s3 , ur_s4]

            int_t_p_1 = convert_to_time_intervals(int_p_1 , w_p , e_p , delta_p , P_p , theta_p)

            int_t_p_2 = convert_to_time_intervals(int_p_2 , w_p , e_p , delta_p , P_p , theta_p)

            int_t_s_1 = convert_to_time_intervals(int_s_1 , w_s , e_s , delta_s , P_s , theta_s)

            int_t_s_2 = convert_to_time_intervals(int_s_2 , w_s , e_s , delta_s , P_s , theta_s)


            intersection = compute_intersection_intervals(int_t_p_1 , int_t_p_2 , int_t_s_1 , int_t_s_2 , rev_p_integer , rev_s_integer, P_p , P_s)
            if intersection == []:
                print(f"Object {ID_2} is filtered out by the time filter")
                filtered_ids.append(ID_2)
            else:
                print(f"Object {ID_2} passes the time filter!")
                print(f"Intersection (seconds) at {intersection}")

    perc_of_filter = (len(filtered_ids))/(len(IDs)) * 100
    non_filtered_ids = [ID for ID in IDs if ID not in filtered_ids]
    print(f"Filtering completed for the datased. About {perc_of_filter}% of objects have been discarted")
    print(f"Number of remaining objects: {len(non_filtered_ids)}")
    print(f"Objects survived: {non_filtered_ids}")

    return non_filtered_ids

# Check for intersection between the two intervals
def intervals_intersect(interval1, interval2):
    return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])

def compute_intersection_intervals(intervalp1 ,intervalp2 , intervals1, intervals2, n_rev_p, n_rev_s, P_p, P_s):
    """
    Compute the intersection intervals between two orbital objects over multiple revolutions.

    Parameters:
    interval1 : tuple
        Time interval for the first object (start, end) in seconds.
    interval2 : tuple
        Time interval for the second object (start, end) in seconds.
    n_rev_p : int
        Number of revolutions for the primary object.
    n_rev_s : int
        Number of revolutions for the secondary object.
    P_p : float
        Orbital period of the primary object in seconds.
    P_s : float
        Orbital period of the secondary object in seconds.

    Returns:
    list
        A list of tuples representing the intersection intervals in seconds.
    """
    intersections = []

    # Generate intervals for the primary object over multiple revolutions
    primary_intervals = []
    for i in range(n_rev_p):
        start1 = intervalp1[0] + i * P_p
        end1 = intervalp1[1] + i * P_p
        primary_intervals.append((start1, end1))
        start2 = intervalp2[0] + i * P_p
        end2 = intervalp2[1] + i * P_p
        primary_intervals.append((start2, end2))

    # Generate intervals for the secondary object over multiple revolutions
    secondary_intervals = []
    for j in range(n_rev_s):
        start1 = intervals1[0] + j * P_s
        end1 = intervals1[1] + j * P_s
        secondary_intervals.append((start1, end1))
        start2 = intervals2[0] + j * P_s
        end2 = intervals2[1] + j * P_s
        secondary_intervals.append((start2, end2))
    # Check for intersections between all intervals
    for p_interval in primary_intervals:
        for s_interval in secondary_intervals:
            if intervals_intersect(p_interval, s_interval):
                intersection_start = max(p_interval[0], s_interval[0])
                intersection_end = min(p_interval[1], s_interval[1])
                intersections.append((intersection_start, intersection_end))

    # If no intersections are found, return an empty list
    if not intersections:
        return []

    return intersections

def compute_K(I_s, I_p, Omega_s, Omega_p):
    """
    Compute the vector K as the cross product of w_s and w_p.

    Parameters:
    I_s : float
        Inclination of the secondary orbit in radians.
    I_p : float
        Inclination of the primary orbit in radians.
    Omega_s : float
        Right ascension of the ascending node (RAAN) of the secondary orbit in radians.
    Omega_p : float
        Right ascension of the ascending node (RAAN) of the primary orbit in radians.

    Returns:
    numpy.ndarray
        The vector K as a 3D numpy array.
    """
    # Compute w_s
    w_s = np.array([
        np.sin(Omega_s) * np.sin(I_s),
        np.cos(Omega_s) * np.sin(I_s),
        np.cos(I_s)
    ])

    # Compute w_p
    w_p = np.array([
        np.sin(Omega_p) * np.sin(I_p),
        np.cos(Omega_p) * np.sin(I_p),
        np.cos(I_p)
    ])

    # Compute K as the cross product of w_s and w_p
    K = np.cross(w_s, w_p)

    return K






def compute_I_r(I_p, I_s, Omega_p, Omega_s):
    """
    Compute the reference inclination I_r based on the given parameters.

    Parameters:
    I_p : float
        Inclination of the primary orbit in radians.
    I_s : float
        Inclination of the secondary orbit in radians.
    Omega_p : float
        Right ascension of the ascending node (RAAN) of the primary orbit in radians.
    Omega_s : float
        Right ascension of the ascending node (RAAN) of the secondary orbit in radians.

    Returns:
    float
        The reference inclination I_r in radians.
    """
    # Compute the magnitude of the vector K
    K = compute_K(I_s, I_p, Omega_s, Omega_p)
    I_r = np.arcsin(np.linalg.norm(K))


    return I_r





def compute_delta(I_p, I_s, Omega_p, Omega_s):
    """
    Compute the delta angles (Δ_p and Δ_s) based on the given parameters.

    Parameters:
    I_p : float
        Inclination of the primary orbit in radians.
    I_s : float
        Inclination of the secondary orbit in radians.
    I_r : float
        Reference inclination in radians.
    Omega_p : float
        Right ascension of the ascending node (RAAN) of the primary orbit in radians.
    Omega_s : float
        Right ascension of the ascending node (RAAN) of the secondary orbit in radians.

    Returns:
    tuple
        A tuple containing Δ_p and Δ_s in radians.
    """
    # Compute Ir 
    I_r = compute_I_r(I_p , I_s , Omega_p , Omega_s)
    # Compute Δ_p
    cos_delta_p = (1 / np.sin(I_r)) * (np.sin(I_p) * np.cos(I_s) - np.sin(I_s) * np.cos(I_p) * np.cos(Omega_p - Omega_s))
    sin_delta_p = (1 / np.sin(I_r)) * (np.sin(I_s) * np.sin(Omega_p - Omega_s))
    delta_p = np.arctan2(sin_delta_p, cos_delta_p)

    # Compute Δ_s
    cos_delta_s = (1 / np.sin(I_r)) * (np.sin(I_p) * np.cos(I_s) * np.cos(Omega_p - Omega_s) - np.sin(I_s) * np.cos(I_p))
    sin_delta_s = (1 / np.sin(I_r)) * (np.sin(I_p) * np.sin(Omega_p - Omega_s))
    delta_s = np.arctan2(sin_delta_s, cos_delta_s)

    return delta_p, delta_s

def compute_a_vectors(e, omega, delta):
    """
    Compute the a_x and a_y vectors based on the given parameters.

    Parameters:
    e : float
        Eccentricity of the orbit.
    omega : float
        Argument of periapsis in radians.
    delta : float
        Delta angle in radians.

    Returns:
    tuple
        A tuple containing a_x and a_y.
    """
    a_x = e * np.cos(omega - delta)
    a_y = e * np.sin(omega - delta)

    return a_x, a_y

def compute_u_r(a, e, I_r, a_x, a_y, D):
    """
    Compute the angle u_r based on the given parameters.

    Parameters:
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.
    I_r : float
        Reference inclination in radians.
    a_x : float
        x-component of the eccentricity vector.
    a_y : float
        y-component of the eccentricity vector.
    D : float
        Distance parameter.

    Returns:
    float
        The angle u_r in radians.
    """
    alpha = a * (1 - e**2) * np.sin(I_r)
    Q = alpha * (alpha - 2 * D * a_y) - (1 - e**2) * D**2
    

    numerator_1 = -D**2 * a_x + (alpha - D * a_y) * np.sqrt(Q)
    numerator_2 = -D**2 * a_x - (alpha - D * a_y) * np.sqrt(Q)
    denominator = alpha * (alpha - 2 * D * a_y) + D**2 * e**2

    cos_u_r_1 = numerator_1 / denominator
    cos_u_r_2 = numerator_2 / denominator
    if Q < 0 or np.abs(cos_u_r_1)>1 or np.abs(cos_u_r_2)>1:
        print(f"Careful! Satellite path does not even get further than {D}m from the other's satellite orbital plane. Treating pair as coplanar... ")
        return -30, -1, -1, -1  # Condition of coplanarity
    
    u_r_1 = np.arccos(cos_u_r_1)  # Ensure the value is within valid range for arccos
    u_r_2 = 2 * np.pi - u_r_1  # Second solution for u_r_1

    u_r_3 = np.arccos(cos_u_r_2)  # Ensure the value is within valid range for arccos
    u_r_4 = 2 * np.pi - u_r_3  # Second solution for u_r_2

    return u_r_1, u_r_2, u_r_3, u_r_4

def convert_to_time_intervals(angular_interval, omega, e, delta, P, theta0):
    """
    Convert an angular interval to time intervals based on orbital parameters.

    Parameters:
    angular_interval : tuple
        A tuple containing the start and end of the angular interval (in radians).
    omega : float
        Argument of periapsis in radians.
    e : float
        Orbital eccentricity.
    delta : float
        Delta angle in radians.
    P : float
        Orbital period in seconds.
    M0 : float
        Mean anomaly at epoch in radians.

    Returns:
    tuple
        A tuple containing the start and end of the time interval (in seconds).
    """
    ur_0, ur_1 = angular_interval

    # Convert angular interval to true anomalies
    true_an_0 = ur_0 - omega + delta
    true_an_1 = ur_1 - omega + delta

    # Convert true anomalies to mean anomalies
    mean_0 = true_to_mean_anomaly(true_an_0, e)
    mean_1 = true_to_mean_anomaly(true_an_1, e)
    M0 = true_to_mean_anomaly(theta0 , e)

    # Compute time intervals using Kepler's equation, adding 10 s for it to be conservative !
    t_0 = (mean_0 - M0) * P / (2 * np.pi) - 10
    t_1 = (mean_1 - M0) * P / (2 * np.pi) + 10

    return t_0, t_1

def true_to_mean_anomaly(true_anomaly, eccentricity):
    """
    Convert true anomaly to mean anomaly using Kepler's equation.

    Parameters:
    true_anomaly : float
        True anomaly in radians.
    eccentricity : float
        Orbital eccentricity.

    Returns:
    float
        Mean anomaly in radians.
    """
    # Compute the eccentric anomaly
    eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(true_anomaly / 2))

    # Ensure eccentric anomaly is in the range [0, 2π]
    if eccentric_anomaly < 0:
        eccentric_anomaly += 2 * np.pi

    # Compute the mean anomaly
    mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)

    return mean_anomaly


def mahalanobis_distance(X1, P1, X2, P2):
    """
    Compute the Mahalanobis distance between two objects.

    Parameters:
    X1 : numpy.ndarray
        State vector of the first object (position and velocity).
    P1 : numpy.ndarray
        Covariance matrix of the first object.
    X2 : numpy.ndarray
        State vector of the second object (position and velocity).
    P2 : numpy.ndarray
        Covariance matrix of the second object.

    Returns:
    float
        The Mahalanobis distance.
    """
    relative_position = X1[:3] - X2[:3]
    combined_covariance = P1[:3, :3] + P2[:3, :3]
    inv_combined_covariance = np.linalg.inv(combined_covariance)
    distance = np.sqrt(relative_position.T @ inv_combined_covariance @ relative_position)
    return distance
def conjunction_assessment(rso_catalog, D, ID , padding = 30e3, treshold = 5e3):

    ### FILTERING OF THE CATALOG ###

    F1_ids = perigee_apogee_filter(rso_catalog , padding , ID)

    N = len(F1_ids)

    filtered_rso_catalog = {key: rso_catalog[key] for key in F1_ids}

    ### SOME objects will survive. Procede computing the TCA(s) using a spherical screening volume...the latter may be output of a function (?)
        
    rso_catalog = filtered_rso_catalog
    ids = rso_catalog.keys()
    state_ref = rso_catalog[ID]['state']
    P_ref = rso_catalog[ID]['covar']
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
        max_step = 1000,
        min_step = 1e-3,
        rtol = 1e-12,
        atol = 1e-12
    )
    T = np.zeros((100, 100)) 
    rho = np.zeros((100, 100))

    t_range = [tdb_epoch , tdb_epoch + 2*constants.JULIAN_DAY]
    # for i in range(len(rso_catalog)):
    #     if list(ids)[i] != ID:
    #         print(i)
    #         id = list(ids)[i]
    #         state_2 = rso_catalog[id]['state']

    #         Cd_2 = rso_catalog[id]['Cd']

    #         Cr_2 = rso_catalog[id]['Cr']

    #         area_2 = rso_catalog[id]['area']

    #         mass_2 = rso_catalog[id]['mass']

    #         state_params_2 = dict(
    #         central_bodies = central_bodies , 
    #         bodies_to_create = bodies_to_create , 
    #         mass = mass_2 , area = area_2 , 
    #         Cd = Cd_2 , Cr = Cr_2 , 
    #         sph_deg = sph_deg , 
    #         sph_ord = sph_ord
    #         )
    #         T_list , rho_list = ConjunctionUtilities.compute_TCA(state_ref, state_2 , t_range, state_params_1 , state_params_2 , int_params, rho_min_crit = D)
    #         n = len(T_list)
    #         T[i, 0:n ] = T_list
    #         rho[i, 0:n ] = rho_list

    # #Save the results to a .dat file
    # with open('T_rho_results_1.dat', 'w') as file:
    #     for i in range(T.shape[0]):
    #         for j in range(T.shape[1]):
    #             if T[i, j] != 0 or rho[i, j] != 0:  # Avoid saving uninitialized values
    #                 file.write(f"{i}\t{j}\t{T[i, j]}\t{rho[i, j]}\n")

    # Of all the different TCA, save the index that violates the treshold, and print their value...

    # Loop through the indices of violating objects
    # Read the file T_rho_results_1.dat and retrieve T and rho
    T = np.zeros((N, 30))
    rho = np.zeros((N, 30))

    try:
        with open('T_rho_results_1.dat', 'r') as file:
            for line in file:
                i, j, T_val, rho_val = map(float, line.split())
                i, j = int(i), int(j)
                T[i, j] = T_val
                rho[i, j] = rho_val
    except FileNotFoundError:
        print("File T_rho_results_1.dat not found.")
    result = []
    for i in range(N):
        Pc_tot = []
        Uc_tot = []
        euclidean_dist = []
        mahalanobius_dist = []
        if any(rho[i, :] < treshold):  # Check if any rho value for this object is below the threshold
            violating_id = list(ids)[i]
            for j in range(T.shape[1]):
                if rho[i, j] != 0 and rho[i, j] < treshold:  # Check if this specific rho value violates the threshold
                    encounter_time = T[i, j]
                    ## Lets compute some collision probability! 
                    ## First, we need propagate everything again to TCA, this time with also the covariance!
                    state_2 = rso_catalog[violating_id]['state']
                    P_2 = rso_catalog[violating_id]['covar']
                    Cd_2 = rso_catalog[violating_id]['Cd']
                    Cr_2 = rso_catalog[violating_id]['Cr']
                    area_2 = rso_catalog[violating_id]['area']
                    mass_2 = rso_catalog[violating_id]['mass']

                    state_params_2 = dict(
                    central_bodies = central_bodies , 
                    bodies_to_create = bodies_to_create , 
                    mass = mass_2 , area = area_2 , 
                    Cd = Cd_2 , Cr = Cr_2 , 
                    sph_deg = sph_deg , 
                    sph_ord = sph_ord
                    )
                    print(f"{i} and {j}")
                    tf, Xf_1 , Pf_1 = TudatPropagator.propagate_state_and_covar(state_ref , P_ref , [tdb_epoch , encounter_time], state_params_1 , int_params)

                    tf, Xf_2 , Pf_2 = TudatPropagator.propagate_state_and_covar(state_2 , P_2 , [tdb_epoch , encounter_time], state_params_2 , int_params)


                    r1 = np.sqrt(area_1/(4*np.pi))

                    r2 = np.sqrt(area_2/(4*np.pi))

                    HBR = r1 + r2

                    Pc_i = ConjunctionUtilities.Pc2D_Foster(Xf_1 , Pf_1, Xf_2 , Pf_2 , HBR)
                    Uc2D_i = ConjunctionUtilities.Uc2D(Xf_1 , Pf_1, Xf_2 , Pf_2 , HBR)
                    distance_at_tca = np.linalg.norm(Xf_1[:3] - Xf_2[:3])
                    mahalanobius_dist_at_tca = mahalanobis_distance(Xf_1 , Pf_1,Xf_2 , Pf_2)
                    Pc_tot.append(Pc_i)
                    Uc_tot.append(Uc2D_i)
                    euclidean_dist.append(distance_at_tca)
                    mahalanobius_dist.append(mahalanobius_dist_at_tca)
            results_dict = {
                "id": violating_id,
                "Pc": Pc_tot,
                "distance_at_tca": euclidean_dist,
                "mahalanobis_distance": mahalanobius_dist,
                "Uc":Uc_tot
            }
            result.append(results_dict)
    return result

