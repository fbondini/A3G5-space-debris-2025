###############################################################################
# This file contains a series of functions used to filter the population of RSO
###############################################################################

import numpy as np
import math
from datetime import datetime
from scipy.integrate import dblquad
from scipy.special import erfcinv
from scipy.optimize import fsolve
import pickle
import time
import os
import matplotlib.pyplot as plt
from tudatpy import astro
from tudatpy import constants

import TudatPropagator as prop

def perigee_apogee_filter(rso_catalog, D, ID_ref):
    print("Start of the perigee-apogee screening")
    filtered_ids = []
    IDs = rso_catalog.keys()
    print(f"Starting population dimension: {len(IDs)}")
    # Cycle within the different objects
    state_ref = rso_catalog[ID_ref]['state']

    kepler = astro.element_conversion.cartesian_to_keplerian(state_ref, 3.986004415e14)

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

            ur_s1 , ur_s2 , ur_s3 , ur_s4 = compute_u_r(a_p , e_p, I_r , ax_p , ay_p , D)

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
    indices_non_filtered = [list(IDs).index(ID) for ID in non_filtered_ids]
    print(f"Indices of non-filtered objects: {indices_non_filtered}")

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