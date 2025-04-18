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
from Testing import*

import TudatPropagator 

# Constants
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
Re = 6378137  # Earth's radius (m)
omega_e = 7.2921159e-5  # Earth's rotation rate (rad/s)
from scipy.constants import g as g0, R as R_universal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
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
    return densities[layer]



def height_from_radius(r):
    """
    Calculates the height above the Earth's surface given the radial distance.
    """
    Re_equator = 6378000  # Earth's equatorial radius (m)
   
    h = r - Re

    return h


def velocity(a, e, v, i):
    """
    Computes the satellite velocity relative to the atmosphere.
    """
    p = a * (1 - e**2)
    n = np.sqrt(mu/a**3)
    V = np.sqrt(mu * (1 + e**2 + 2 * e * np.cos(v)) / p) *(1 - ((1-e**2)**(3/2))/(1+e**2 + 2*e*np.cos(v))*(omega_e/n)*np.cos(i))
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

    integrand = lambda v: atmospheric_density(a , e, v) * velocity(a, e, v , i) * (
        1 + e**2 + 2 * e * np.cos(v) - omega_e * np.cos(i) * np.sqrt(
        (a**3 * (1 - e**2)**3) / mu) )* (r(a, e, v)**2 / (a * (1 - e**2)**(3/2)))
    result, _ = quad(integrand, 0, 2 * np.pi)
    return -B / (2 * np.pi) * result


def de_dt(a, e, i, B):
    """
    Computes the rate of change of the eccentricity (de/dt) due to atmospheric drag.
    """
    integrand = lambda v: atmospheric_density(a,e, v) * velocity(a, e, v , i) * (
        e + np.cos(v) - (r(a, e, v)**2 * omega_e * np.cos(i)) / (2 * np.sqrt(mu * a * (1 - e**2))) * (2 * (e + np.cos(v)) - e * np.sin(v)**2)) *(
        (r(a, e, v) / a)**2 * (1 - e**2)**(-1/2))
    result, _ = quad(integrand, 0, 2 * np.pi)
    return -B / (2 * np.pi) * result



def J2_J3_perturbations(a, e, i , omega):
    """
    Computes the J2 and J3 perturbation effects on the rates of change of the orbital elements.
    """
    J2 = 1.7553e-5
    J3 = -2.619e-7
    p = a * (1 - e**2)
    n = np.sqrt(mu/a**3)

    # Perturbation terms due to J2
    dOmega_J2 = -3/2 *n* J2 * (Re / p)**2 * np.cos(i) # Ok

    domega_J2 = 3/4 * n* J2 * (Re / p)**2 * (4 - 5 * np.sin(i)**2) # ok

    # Perturbation terms due to J3
    dOmega_J3 = 0 # ok
    domega_J3 = 3/8 * n* J3 * (Re / p)**3 *(  (4 - 5 * np.sin(i)**2)* ((np.sin(i) - e**2*np.cos(i)**2)/(e*np.sin(i))) + 2*np.sin(i)*(13-15*np.sin(i)**2)*e)*np.sin(omega)  #Ok
    de_J3 = -3/8*n*J3*(Re/p)**3*np.sin(i)*(4-5*np.sin(i)**2)*(1-e**2)*np.cos(omega)

    dM_J2 = 3/4 * J2 * n * (Re/p)**2*( 1 - e**2)*(3*np.cos(i)**2 -1)
    dM_J3 =0# -(3/8) * n * J3 * (Re / p)**3 * np.sin(i) * (4 - 5 * np.sin(i)**2) * (1 - 4 * e**2) * ((1 - e**2)**(1/2) / e) * np.sin(omega)
    dM_dt = dM_J2 + dM_J3

    return dOmega_J2 + dOmega_J3, domega_J2 + domega_J3 , de_J3, dM_dt


def orbital_elements_rate(a, e, i, omega, B):
    """
    Calculates the rate of change of all orbital elements.
    """
    da = da_dt(a, e, i, B)
    de = de_dt(a, e, i, B)
    di = 0  # Inclination rate of change is zero for the simplified model
    dOmega_J2_J3, domega_J2_J3 , de_J3, dM = J2_J3_perturbations(a, e, i, omega)
    dOmega = dOmega_J2_J3
    domega =  domega_J2_J3

    de = de + de_J3

    return da, de, di, dOmega, domega, dM




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

    da, de, di, dOmega, domega , dM= orbital_elements_rate(kepler[0], kepler[1], kepler[2], kepler[4],B_ref)

    print(f"Variation in semi-major axis (da): {da * 2 * constants.JULIAN_DAY}")
    print(f"Variation in eccentricity (de): {de * 2 * constants.JULIAN_DAY}")
    print(f"Variation in inclination (di): {di * 2 * constants.JULIAN_DAY}")
    print(f"Variation in RAAN (dOmega): {dOmega * 2 * constants.JULIAN_DAY}")
    print(f"Variation in argument of perigee (domega): {domega * 2 * constants.JULIAN_DAY}")

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
    Cdp = rso_catalog[ID_ref]['Cd']
    Ap = rso_catalog[ID_ref]['area']
    mp= rso_catalog[ID_ref]['mass']
    B_p = Cdp*(Ap/mp)

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
            Cd = rso_catalog[ID_2]['Cd']
            A = rso_catalog[ID_2]['area']
            m= rso_catalog[ID_2]['mass']
            B_s = Cd*(A/m)
            kepler = astro.element_conversion.cartesian_to_keplerian(state, 3.986004415e14)
            print(f"COE retreived")
            a_s = kepler[0]
            e_s = kepler[1]
            I_s = kepler[2]
            w_s = kepler[3]
            Omega_s = kepler[4]
            theta_s = kepler[5]
            P_s = 2*np.pi*np.sqrt(a_s**3/ 3.986004415e14)
            filtered_id = process_object(ID_2 , a_p , e_p , I_p , Omega_p , w_p , theta_p , P_p , B_p, D , a_s , e_s , I_s , Omega_s , w_s , theta_s , P_s , B_s)
            if filtered_id !=[]:
                filtered_ids.append(filtered_id)

    perc_of_filter = (len(filtered_ids))/(len(IDs)) * 100
    non_filtered_ids = [ID for ID in IDs if ID not in filtered_ids]
    print(f"Filtering completed for the datased. About {perc_of_filter}% of objects have been discarted")
    print(f"Number of remaining objects: {len(non_filtered_ids)}")
    print(f"Objects survived: {non_filtered_ids}")

    return non_filtered_ids



def process_object(ID_2, a_p, e_p, I_p, Omega_p, w_p, theta_p, P_p, B_p, D, a_s, e_s, I_s, Omega_s, w_s, theta_s, P_s, B_s):
    """
    Process a single object for the time filter.

    Parameters:
    ID_2 : str
        ID of the object being processed.
    a_p, e_p, I_p, Omega_p, w_p, theta_p, P_p, B_p : float
        Orbital parameters of the primary object.
    D : float
        Distance parameter.
    a_s, e_s, I_s, Omega_s, w_s, theta_s, P_s, B_s : float
        Orbital parameters of the secondary object.
    filtered_ids : list
        List to store IDs of filtered objects.

    Returns:
    None
    """
    rev_s = (2 * constants.JULIAN_DAY) / P_s
    rev_s_integer = int(rev_s)
    rev_s_decimal = rev_s - rev_s_integer

    I_r = compute_I_r(I_p, I_s, Omega_p, Omega_s)
    print(f"Relative inclination of {np.rad2deg(I_r)} degree")

    delta_p, delta_s = compute_delta(I_p, I_s, Omega_p, Omega_s)

    ax_p, ay_p = compute_a_vectors(e_p, w_p, delta_p)
    ax_s, ay_s = compute_a_vectors(e_s, w_s, delta_s)

    ur_p1, ur_p2, ur_p3, ur_p4 = compute_u_r(a_p, e_p, I_r, ax_p, ay_p, D)
    ur_s1, ur_s2, ur_s3, ur_s4 = compute_u_r(a_s, e_s, I_r, ax_s, ay_s, D)

    if ur_p1 == -30 or ur_s1 == -30:
        print(f"Object {ID_2} passes the time filter!, But just because it is coplanar")
        return []
    else:
        int_p_1 = [ur_p1, ur_p2]
        int_p_2 = [ur_p3, ur_p4]
        int_s_1 = [ur_s1, ur_s2]
        int_s_2 = [ur_s3, ur_s4]

        int_t_p_1 = convert_to_time_intervals(int_p_1, w_p, e_p, delta_p, P_p, theta_p)
        int_t_p_2 = convert_to_time_intervals(int_p_2, w_p, e_p, delta_p, P_p, theta_p)
        int_t_s_1 = convert_to_time_intervals(int_s_1, w_s, e_s, delta_s, P_s, theta_s)
        int_t_s_2 = convert_to_time_intervals(int_s_2, w_s, e_s, delta_s, P_s, theta_s)

        intersection = compute_intersection_intervals(
            int_t_p_1, int_t_p_2, int_t_s_1, int_t_s_2,P_p, P_s, a_p, e_p, I_p, Omega_p, w_p, B_p, a_s, e_s, I_s, Omega_s, w_s, B_s, delta_p, delta_s
        )
        if intersection == []:
            print(f"Object {ID_2} is filtered out by the time filter")
            return ID_2
        else:
            print(f"Object {ID_2} passes the time filter!")
            print(f"Intersection (seconds) at {intersection}")
            return []
        
















# Check for intersection between the two intervals
def intervals_intersect(interval1, interval2):
    return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])

def compute_intersection_intervals(intervalp1 ,intervalp2 , intervals1, intervals2, P_p, P_s , ap , ep , ip ,O_p , wp, B_p,
                                    a_s , e_s , i_s ,O_s , w_s, B_s , delta_p , delta_s):
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
    da , de, di , dO, dw , dM = orbital_elements_rate(ap , ep , ip , wp, B_p)
    das , des, dis , dOs, dws , dMs = orbital_elements_rate(a_s, e_s , i_s , w_s, B_s)

    delta_p_dot , delta_s_dot = compute_delta_dot(ip, i_s, delta_p , delta_s , dO , dOs , O_p , O_s)
    n0p =2* np.pi/P_p
    n0s = 2*np.pi/P_s 


    PDF_p = compute_PDF(n0p , dM, dw , delta_p_dot)
    PDF_s = compute_PDF(n0s , dMs , dws, delta_s_dot)
    dn_p = compute_n_dot(da , 3.986004415e14 , ap)
    dn_s = compute_n_dot(das , 3.986004415e14 , a_s)

    i = 0
    j = 0
    time_limit = 2*constants.JULIAN_DAY
    stay = True
    # Generate intervals for the primary object over multiple revolutions
    primary_intervals = []
    while stay:
        i = i +1
        ### Recompute all the orbital parameters 
        P_p = (2*np.pi)/(n0p + dM + dw - delta_p_dot + i*PDF_p*dn_p)#PDF_p*(1-(2*np.pi/n0p)*(dn_p/n0p)*i)
        start1 = intervalp1[0] + i * P_p
        end1 = intervalp1[1] + i * P_p
        start2 = intervalp2[0] + i * P_p
        end2 = intervalp2[1] + i * P_p
        if start1 < time_limit and end1 < time_limit and start2 < time_limit and end2 < time_limit:
            primary_intervals.append((start1, end1))
            primary_intervals.append((start2, end2))
        else:
            stay = False
    stay = True
    # Generate intervals for the secondary object over multiple revolutions
    secondary_intervals = []
    while stay:
        j = j+1
        P_s = (2*np.pi)/(n0s + dMs + dws - delta_s_dot + j*PDF_s*dn_s)
        start1 = intervals1[0] + j * P_s
        end1 = intervals1[1] + j * P_s
        start2 = intervals2[0] + j * P_s
        end2 = intervals2[1] + j * P_s
        if start1 < time_limit and end1 < time_limit and start2 < time_limit and end2 < time_limit:
            secondary_intervals.append((start2, end2))
            secondary_intervals.append((start1, end1))
        else: 
            stay = False
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

def compute_n_dot(da , mu, a ):
    return -(3/2)*(mu/a**4)*(mu/a**3)**(-1/2)*da
def compute_modified_period(P_DF, n0_dot, n0, K):
    """
    Compute the modified period P_K based on the given parameters.

    Parameters:
    P_DF : float
        The nominal period of the orbit.
    n0_dot : float
        The rate of change of the mean motion.
    n0 : float
        The mean motion.
    K : float
        A constant factor.

    Returns:
    float
        The modified period P_K.
    """
    P_K = P_DF * (1 - (2 * np.pi * n0_dot * K) / (n0**2))
    return P_K

def compute_PDF(n0, M0_dot , w0_dot , delta0_dot):
    PDF = (2*np.pi)/(n0 + M0_dot + w0_dot - delta0_dot)
    return PDF






def compute_delta_dot(I_p, I_s, delta_p, delta_s, Omega_dot_p, Omega_dot_s , Omega_p , Omega_s):
    """
    Compute the rate of change of the delta angles (Δ_p and Δ_s) based on the given parameters.

    Parameters:
    I_p : float
        Inclination of the primary orbit in radians.
    I_s : float
        Inclination of the secondary orbit in radians.
    I_r : float
        Reference inclination in radians.
    Omega_dot_p : float
        Rate of change of RAAN for the primary orbit in radians per second.
    Omega_dot_s : float
        Rate of change of RAAN for the secondary orbit in radians per second.

    Returns:
    tuple
        A tuple containing the rates of change of Δ_p and Δ_s in radians per second.
    """
    I_r = compute_I_r(I_p , I_s , Omega_p, Omega_s)
    delta_dot_p = (1 / np.sin(I_r)) * np.sin(I_s) * np.cos(delta_s)*(Omega_dot_p - Omega_dot_s)

    delta_dot_s = (1 / np.sin(I_r)) * np.sin(I_p)*np.cos(delta_p)*(Omega_dot_p - Omega_dot_s)

    return delta_dot_p, delta_dot_s




def compute_K(I_s, I_p, Omega_s, Omega_p):
    """
    Compute the vector K as the cross product of w_s and w_p.

    Parameters:
    I_s : float
        Inclination of the secondary orbit in radians.
    I_p : float
        Inclination of the primary orbit in radians.
    Omega_s : float
        RAAN of the secondary orbit in radians.
    Omega_p : float
        RAAN of the primary orbit in radians.

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
    ur_0 = angular_interval[0]

    ur_1 = angular_interval[1]

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
def conjunction_assessment(rso_catalog, ID , padding = 30e3, treshold = 5e3, original_epoch =  796780800.0):
    """
    Process the RSO catalog file, providing filtering, screening, CDMs and HIEs of 
    possible conjunctions with a given reference ID. The conjunction_assessment function first 
    filters the catalogue using a simple apogee-perigee filter, then uses an elliptical screening volume to further reduce the
    RSOs. Then a treshold of 5km is imposed to determine the possible CDMs, and, if other condtions are respected, of HIEs.

    Parameters: 
        rso_catalog : dict

        ID : int

        padding = 30e3 : pad used in the apogee-perigee filter

        treshold = 5e3 : spherical volume used for CDMs detection
    
    Returns:

        result: dict of all the HIEs
    """


    # ### FILTERING OF THE CATALOG ###

    F1_ids = perigee_apogee_filter(rso_catalog , padding , ID)

    filtered_rso_catalog = {key: rso_catalog[key] for key in F1_ids}

    N = len(F1_ids)

    # F2_ids = time_filter(filtered_rso_catalog , D , ID)

    # filtered_rso_catalog_2 = {key: filtered_rso_catalog[key] for key in F2_ids}

    # N = len(F2_ids)

    F2_ids = screening_volume(filtered_rso_catalog , ID )

    filtered_rso_catalog_2 = {key: filtered_rso_catalog[key] for key in F2_ids}

    N = len(F2_ids)

    print(f"Number of objects after filtering: {N}")

    ### SOME objects will survive. Procede computing the TCA(s) using a spherical screening volume...the latter may be output of a function (?)

    rso_catalog = filtered_rso_catalog_2
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
    int_params2 = dict(
        tudat_integrator = 'rkf78',
        step = -10,
        max_step = 1000,
        min_step = 1e-3,
        rtol = 1e-12,
        atol = 1e-12
    )
    T = np.zeros((100, 100)) 
    rho = np.zeros((100, 100))
    Id = np.zeros(100)

    t_range = [tdb_epoch , original_epoch + 2*constants.JULIAN_DAY]
    for i in range(len(rso_catalog)):
        if list(ids)[i] != ID:
            print(i)
            id = list(ids)[i]
            state_2 = rso_catalog[id]['state']

            Cd_2 = rso_catalog[id]['Cd']

            Cr_2 = rso_catalog[id]['Cr']

            area_2 = rso_catalog[id]['area']

            mass_2 = rso_catalog[id]['mass']

            tdb_epoch_2 = rso_catalog[id]['epoch_tdb']

            state_params_2 = dict(
            central_bodies = central_bodies , 
            bodies_to_create = bodies_to_create , 
            mass = mass_2 , area = area_2 , 
            Cd = Cd_2 , Cr = Cr_2 , 
            sph_deg = sph_deg , 
            sph_ord = sph_ord
            )
            T_list , rho_list = ConjunctionUtilities.compute_TCA(state_ref, state_2 , t_range, state_params_1 , state_params_2 , int_params, rho_min_crit = 5e3)

            n = len(T_list)
            T[i, 0:n ] = T_list
            rho[i, 0:n ] = rho_list
            Id[i] = id

    # Save the results to a .dat file
    with open('T_rho_results_CMD.dat', 'w') as file:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i, j] != 0 or rho[i, j] != 0:  # Avoid saving uninitialized values
                    file.write(f"{i}\t{j}\t{Id[i]}\t{T[i, j]}\t{rho[i, j]}\n")

    # Initialize T and rho arrays for storing results
    T = np.zeros((N, 30))
    rho = np.zeros((N, 30))

    try:
        # Read the file T_rho_results_CMD.dat and retrieve T, rho, and Id
        with open('T_rho_results_CMD.dat', 'r') as file:
            for line in file:
                i, j, id_val, T_val, rho_val = line.split()
                i, j = int(i), int(j)
                T[i, j] = float(T_val)
                rho[i, j] = float(rho_val)
                Id[i] = float(id_val)
    except FileNotFoundError:
        print("File T_rho_results_CMD.dat not found.")
    result = []
    for i in range(N):
        Pc_tot = []
        Uc_tot = []
        euclidean_dist = []
        mahalanobius_dist = []
        POS = []
        VEL = []
        close_time_TDB = []
        Pf_primary = []
        Pf_secondary = []
        X_f_rel = []
        X1 = []
        X2 = []
        if any(rho[i, :] < treshold):  # Check if any rho value for this object is below the threshold

            violating_id = Id[i]  # Take the Id of the violating object

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

                    tdb_epoch_2 = rso_catalog[violating_id]['epoch_tdb']

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
                    X_f_relative = Xf_2 - Xf_1
                    X_f_pos_RIC = ConjunctionUtilities.eci2ric(Xf_1[:3], Xf_1[3:6], X_f_relative[:3])

                    X_f_vel_RIC = ConjunctionUtilities.eci2ric(Xf_1[:3], Xf_1[3:6], X_f_relative[3:6])

                    Pc_i = ConjunctionUtilities.Pc2D_Foster(Xf_1 , Pf_1, Xf_2 , Pf_2 , HBR)

                    Uc2D_i = ConjunctionUtilities.Uc2D(Xf_1 , Pf_1, Xf_2 , Pf_2 , HBR)

                    distance_at_tca = np.linalg.norm(Xf_1[:3] - Xf_2[:3])
                
                    mahalanobius_dist_at_tca = mahalanobis_distance(Xf_1 , Pf_1, Xf_2 , Pf_2)

                  
                    close_time_TDB.append(encounter_time)

                    UTC_datetime = astro.time_conversion.date_time_from_epoch(encounter_time)

                    year = UTC_datetime.year
                    month = UTC_datetime.month
                    day = UTC_datetime.day
                    hour = UTC_datetime.hour
                    minute = UTC_datetime.minute
                    seconds = UTC_datetime.seconds

                    formatted_date = f"{year:04d}-{month:02d}-{day:02d}"
                    formatted_time = f"{hour:02d}:{minute:02d}:{seconds:06.3f}"

                    # Print the complete calendar date and time in a readable format
                    print(f"Calendar Date and Time: {formatted_date} {formatted_time}")

     


                    Pf_primary.append(Pf_1)

                    Pf_secondary.append(Pf_2)

                    Pc_tot.append(Pc_i)

                    Uc_tot.append(Uc2D_i)

                    euclidean_dist.append(distance_at_tca)

                    mahalanobius_dist.append(mahalanobius_dist_at_tca)

                    POS.append(X_f_pos_RIC)

                    VEL.append(X_f_vel_RIC)

                    X_f_rel.append(X_f_relative)

                    X1.append(Xf_1)
                    X2.append(Xf_2)
                    print("I am here")
                    if np.abs(X_f_pos_RIC[0]) < 200 and mahalanobius_dist_at_tca < 4.6 and distance_at_tca < 1e3:
                        if Uc2D_i > 1e-4:
                            print(f"The object {violating_id} is an HIE, Delande")
                            print(f"Uc = {Uc2D_i}")
                            print(f"Pc = {Pc_i}")
                            print(f"Distance at TCA {distance_at_tca}")
                            print(f"Mahalanobis distance at TCA {mahalanobius_dist_at_tca}")
                            print(f"Close approach TDB : {encounter_time}")
                            print(f"Radial Distance with primary: {X_f_pos_RIC[0]} ")
                            print(f"RIC position: { X_f_pos_RIC}")
                            print(f"RIC velocity: { X_f_vel_RIC}")
                            print(f"Appending results for object {violating_id}...")
                            results_dict = {
                                violating_id: {
                                    "Pc": Pc_tot,
                                    "distance_at_tca": euclidean_dist,
                                    "mahalanobis_distance": mahalanobius_dist,
                                    "Uc": Uc_tot,
                                    "Pos_RIC": POS,
                                    "Vel_RIC": VEL,
                                    "close_time_TDB": close_time_TDB,
                                    "Pf_primary": Pf_primary,
                                    "Pf_secondary": Pf_secondary,
                                    "X_f_rel": X_f_rel,
                                    "X1":X1,
                                    "X2": X2,
                                }
                            }
                            result.append(results_dict)
                        else: 
                            print(f"The object {violating_id} is NOT an HIE, just CDM")
                            print(f"Uc = {Uc2D_i}")
                            print(f"Pc = {Pc_i}")
                            print(f"Distance at TCA {distance_at_tca}")
                            print(f"Close approach TDB : {encounter_time}")
                            print(f"Radial Distance with primary: {X_f_pos_RIC[0]} ")
                            print(f"RIC position: { X_f_pos_RIC}")
                            print(f"RIC velocity: { X_f_vel_RIC}")
                            print(f"Mahalanobis distance at TCA {mahalanobius_dist_at_tca}")

                    elif np.abs(X_f_pos_RIC[0]) < 200 and mahalanobius_dist_at_tca > 4.6 and distance_at_tca < 1e3:
                        if Pc_i > 1e-4:
                            print(f"The object {violating_id} is an HIE, Foster")
                            print(f"Uc = {Uc2D_i}")
                            print(f"Pc = {Pc_i}")
                            print(f"Distance at TCA {distance_at_tca}")
                            print(f"Mahalanobis distance at TCA {mahalanobius_dist_at_tca}")
                            print(f"Close approach TDB : {encounter_time}")
                            print(f"Radial Distance with primary: {X_f_pos_RIC[0]} ")
                            print(f"RIC position: { X_f_pos_RIC}")
                            print(f"RIC velocity: { X_f_vel_RIC}")
                            print(f"Appending results for object {violating_id}...")
                            results_dict = {
                                violating_id: {
                                    "Pc": Pc_tot,
                                    "distance_at_tca": euclidean_dist,
                                    "mahalanobis_distance": mahalanobius_dist,
                                    "Uc": Uc_tot,
                                    "Pos_RIC": POS,
                                    "Vel_RIC": VEL,
                                    "close_time_TDB": close_time_TDB,
                                    "Pf_primary": Pf_primary,
                                    "Pf_secondary": Pf_secondary,
                                    "X_f_rel": X_f_rel,
                                    "X1":X1,
                                    "X2": X2,
                                }
                            }
                            result.append(results_dict)
                        else:
                            print(f"The object {violating_id} is NOT an HIE, just CDM")
                            print(f"Uc = {Uc2D_i}")
                            print(f"Pc = {Pc_i}")
                            print(f"Distance at TCA {distance_at_tca}")
                            print(f"Close approach TDB : {encounter_time}")
                            print(f"Radial Distance with primary: {X_f_pos_RIC[0]} ")
                            print(f"RIC position: { X_f_pos_RIC}")
                            print(f"RIC velocity: { X_f_vel_RIC}")
                            print(f"Mahalanobis distance at TCA {mahalanobius_dist_at_tca}")
                    else: 
                        print(f"The object {violating_id} is NOT an HIE, just CDM")
                        print(f"Uc = {Uc2D_i}")
                        print(f"Pc = {Pc_i}")
                        print(f"Distance at TCA {distance_at_tca}")
                        print(f"Close approach TDB : {encounter_time}")
                        print(f"Radial Distance with primary: {X_f_pos_RIC[0]} ")
                        print(f"RIC position: { X_f_pos_RIC}")
                        print(f"RIC velocity: { X_f_vel_RIC}")
                        print(f"Mahalanobis distance at TCA {mahalanobius_dist_at_tca}")


    return result

def conj_ass_Q2(rso_catalog , ID_ref , ID_maneuver, tdb_epoch_deltav , final_state , final_covariance):
    ids = rso_catalog.keys()
    state_ref = rso_catalog[ID_ref]['state']
    P_ref = rso_catalog[ID_ref]['covar']
    tdb_epoch = rso_catalog[ID_ref]['epoch_tdb']
    Cd_1 = rso_catalog[ID_ref]['Cd']

    Cr_1 = rso_catalog[ID_ref]['Cr']

    area_1 = rso_catalog[ID_ref]['area']

    mass_1 = rso_catalog[ID_ref]['mass']
      # Define additional parameters for the propagation

    sph_deg = 8

    sph_ord = 8

    central_bodies = ['Earth']

    bodies_to_create = ['Earth', 'Sun', 'Moon']

    state_params = dict(
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
    # Propagate the ref body to the epoch of the maneuver

    _ , state_ref_f , P_ref_f = TudatPropagator.propagate_state_and_covar(state_ref , P_ref , [tdb_epoch , tdb_epoch_deltav],state_params , int_params )

    # At this point, construct the dictionary
    rso_catalog_q2 = {
        ID_ref: {
            "state": state_ref_f,
            "covar": P_ref_f,
            "epoch_tdb": tdb_epoch_deltav,
            "area": area_1,
            "mass": mass_1,
            "Cr": Cr_1,
            "Cd": Cd_1,
        },
        ID_maneuver: {
            "state": final_state,
            "covar": final_covariance,
            "epoch_tdb": tdb_epoch_deltav,
            "area": area_1,
            "mass": mass_1,
            "Cr": Cr_1,
            "Cd": Cd_1,
        },
    }

    # Then, simply give it to the conj ass function (should work exactly the same with two objects )

    result = conjunction_assessment(rso_catalog_q2 , ID_ref)

    return result


def conj_ass_Q3(rso_catalog , ID_ref , state_new , covar_new , ID_body, Cd_new =2.2, Cr_new = 1.3 , mass_new = 100 , area_new = 1 ):


    state_ref = rso_catalog[ID_ref]['state']
    P_ref = rso_catalog[ID_ref]['covar']
    tdb_epoch = rso_catalog[ID_ref]['epoch_tdb']
    Cd_1 = rso_catalog[ID_ref]['Cd']

    Cr_1 = rso_catalog[ID_ref]['Cr']

    area_1 = rso_catalog[ID_ref]['area']

    mass_1 = rso_catalog[ID_ref]['mass']

    # Take also the parameters from the Q3 ID
    tdb_epoch_q3 = rso_catalog[ID_body]['epoch_tdb']

    rso_catalog_q3 = {}
    
    # Construct a smaller dictionary of objects 

    rso_catalog_q3[ID_ref] = {
        "state": state_ref,
        "covar": P_ref,
        "epoch_tdb": tdb_epoch,
        "area" : area_1,
        "mass" : mass_1, 
        "Cr" : Cr_1,
        "Cd" : Cd_1,
    }

    rso_catalog_q3[ID_body] = {
        "state": state_new,
        "covar": covar_new,
        "epoch_tdb": tdb_epoch_q3,
        "area" : area_new,
        "mass" : mass_new, 
        "Cr" : Cr_new,
        "Cd" : Cd_new,
    }

    # Then, simply give it to the conj ass function (should work exactly the same with two objects )

    result = conjunction_assessment(rso_catalog_q3, ID_ref)

    plot_3D_orbits(rso_catalog , ID_ref , result)

    return result



def screening_volume(rso_catalog , ID , filtered_ids = [] , original_epoch =  796780800.0):

    ids = rso_catalog.keys()
    state_ref = rso_catalog[ID]['state']
    P_ref = rso_catalog[ID]['covar']
    tdb_epoch = rso_catalog[ID]['epoch_tdb']
    Cd_1 = rso_catalog[ID]['Cd']
    Cr_1 = rso_catalog[ID]['Cr']
    area_1 = rso_catalog[ID]['area']
    mass_1 = rso_catalog[ID]['mass']
    filtered_ids.append(ID)

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
    step = 1
    # Define integrator parameters = dict(step , max_step, min_step,rtol, atol, tudat_integrator)
    int_params = dict(
        tudat_integrator = 'rk4',
        step = 1,
        # max_step = 1000,
        # min_step = 1e-3,
        # rtol = 1e-12,
        # atol = 1e-12
    )
    t_hist , X_hist = TudatPropagator.propagate_orbit(state_ref , [tdb_epoch , original_epoch + 2*constants.JULIAN_DAY], state_params_1, int_params)

    # Now loop for each object in the catalog and compute the propagated history
    for i in range(len(ids)):
        if list(ids)[i] != ID:
            print(f"Analyzing object {list(ids)[i]}...")
            id = list(ids)[i]
            state_2 = rso_catalog[id]['state']

            Cd_2 = rso_catalog[id]['Cd']

            Cr_2 = rso_catalog[id]['Cr']

            area_2 = rso_catalog[id]['area']

            mass_2 = rso_catalog[id]['mass']

            tdb_epoch_2 = rso_catalog[id]['epoch_tdb']

            state_params_2 = dict(
            central_bodies = central_bodies , 
            bodies_to_create = bodies_to_create , 
            mass = mass_2 , area = area_2 , 
            Cd = Cd_2 , Cr = Cr_2 , 
            sph_deg = sph_deg , 
            sph_ord = sph_ord
            )

            int_params = dict(
                tudat_integrator = 'rk4',
                step = 1,
            )

            _, X_hist_2 = TudatPropagator.propagate_orbit(state_2 , [tdb_epoch_2 , original_epoch + 2*constants.JULIAN_DAY], state_params_2, int_params)
            # Compute the relative state vector
            X_rel = X_hist_2 - X_hist
            counter = 0
            # Rotate the vector in RIC frame and check conditions
            for j in range(len(X_rel[:, 0])):
                X_ric = ConjunctionUtilities.eci2ric(X_hist[j, 0:3], X_hist[j, 3:6], X_rel[j, :3])

                ellipsoid_check = (X_ric[0] / 2e3)**2 + (X_ric[1] / 25e3)**2 + (X_ric[2] / 25e3)**2
                if ellipsoid_check < 1:
                    counter += 1
            if counter > 0:
                print(f"The object {id} entered the screening volume for {counter * step} seconds.")
                filtered_ids.append(id)
    # Calculate the number of remaining objects and percentage
    total_objects = len(ids)
    remaining_objects = len(filtered_ids)
    percentage_remaining = (remaining_objects / total_objects) * 100

    print(f"Screening volume completed.")
    print(f"Total objects: {total_objects}")
    print(f"Remaining objects after screening: {remaining_objects}")
    print(f"Percentage of remaining objects: {percentage_remaining:.2f}%")

    return filtered_ids



def processing_results_gaussian(result, rso_full_catalog, ID_ref):
    """
    Process the results of conjunction analysis and visualize the Gaussian distribution of the results.

    Parameters:
    ID : str
        ID of the primary object.
    result : list
        List of dictionaries containing conjunction analysis results.
    rso_full_catalog : dict
        Full catalog of RSOs with their states and covariances.
    ID_ref : str
        Reference ID for the primary object.

    Returns:
    None
    """
    import matplotlib.pyplot as plt


    Pc = []
    distance_at_tca = []
    mahalanobis_distance = []
    Uc = []
    Pos_RIC = []
    Vel_RIC = []
    close_time_TDB = []
    Pf_primary = []
    Pf_secondary = []
    object_ids = []
    X_f_rel = []

    for result_dict in result:
        for obj_id, metrics in result_dict.items():
            print(obj_id)
            object_ids.append(obj_id)
            for key, value in metrics.items():
                if key == "Pc":
                    Pc.append(value)
                elif key == "distance_at_tca":
                    distance_at_tca.append(value)
                elif key == "mahalanobis_distance":
                    mahalanobis_distance.append(value)
                elif key == "Uc":
                    Uc.append(value)
                elif key == "Pos_RIC":
                    Pos_RIC.append(value)
                elif key == "Vel_RIC":
                    Vel_RIC.append(value)
                elif key == "close_time_TDB":
                    close_time_TDB.append(value)
                elif key == "Pf_primary":
                    Pf_primary.append(value)
                elif key == "Pf_secondary":
                    Pf_secondary.append(value)
                elif key == "X_f_rel":
                    X_f_rel.append(value)
                    
    print(Pos_RIC[1])
    print(close_time_TDB)
    for i in range(len(object_ids)):
        print(i)
        violating_id = object_ids[i]
        print(f"Processing object {violating_id}...")

        current_Pc = Pc[i]
        current_distance_at_tca = distance_at_tca[i]
        current_mahalanobis_distance = mahalanobis_distance[i]
        current_Uc = Uc[i]
        current_Pos_RIC = Pos_RIC[i]
        current_Vel_RIC = Vel_RIC[i]
        current_close_time_TDB = close_time_TDB[i]
        current_Pf_primary = Pf_primary[i]
        current_Pf_secondary = Pf_secondary[i]
        X_f_rel_i = X_f_rel[i]

        # Convert lists to numpy arrays for easier manipulation
        Pci = np.array(current_Pc)
        distance_at_tcai = np.array(current_distance_at_tca)
        mahalanobis_distancei = np.array(current_mahalanobis_distance)
        Uci = np.array(current_Uc)
        Pos_RICi = np.array(current_Pos_RIC)
        Vel_RICi = np.array(current_Vel_RIC)
        close_time_TDBi = np.array(current_close_time_TDB)
        Pf_primaryi = np.array(current_Pf_primary)
        Pf_secondaryi = np.array(current_Pf_secondary)
        X_f_reli = np.array(X_f_rel_i)
  
       



        # Compute relative state and covariance in RIC frame
        relative_position = np.array(Pos_RICi).flatten()
        relative_velocity = np.array(Vel_RICi).flatten()

        relative_position_ECI = np.array(X_f_reli[0:3]).flatten()

        relative_velocity_ECI = np.array(X_f_reli[3:6]).flatten()

        combined_covariance = Pf_primaryi[0][:3, :3] + Pf_secondaryi[0][:3, :3]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, color='red', label='Primary Object (Reference Body)')
        ax.scatter(relative_position_ECI[0] , relative_position_ECI[1], relative_position_ECI[2], color='blue', label='Secondary Object (Relative Position)')

        # Plot the ellipsoid centered at the reference body
        plot_sigma_ellipsoid(ax, Pf_primaryi[0][:3, :3], np.zeros(3), label='Uncertainty Ellipsoid', color='red')
        plot_sigma_ellipsoid(ax, Pf_secondaryi[0][:3, :3], relative_position_ECI, label='Uncertainty Ellipsoid', color='cyan')


        

        # Plot 3D ellipsoid for the combined covariance centered at the reference body
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, color='red', label='Primary Object (Reference Body)')
        ax.scatter(relative_position[0] , relative_position[1], relative_position[2], color='blue', label='Secondary Object (Relative Position)')

        # Plot the ellipsoid centered at the reference body
        plot_sigma_ellipsoid(ax, combined_covariance, np.zeros(3), label='Uncertainty Ellipsoid', color='cyan')

        # Plot a cylinder along the relative velocity vector
        relative_velocity_unit = relative_velocity / np.linalg.norm(relative_velocity)
        cylinder_length = 10000  # Length of the cylinder in meters
        cylinder_radius = np.sqrt(100 / np.pi)  # Radius of the cylinder (A = 1)

        # Generate cylinder points
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(-cylinder_length / 2, cylinder_length / 2, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cylinder_radius * np.cos(theta_grid)
        y_grid = cylinder_radius * np.sin(theta_grid)

        # Rotate and translate the cylinder to align with the relative velocity vector
        cylinder_points = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        rotation_matrix = np.linalg.svd(np.outer(relative_velocity_unit, [0, 0, 1]))[0]
        rotated_cylinder_points = rotation_matrix @ cylinder_points
        x_rotated = rotated_cylinder_points[0, :].reshape(x_grid.shape)
        y_rotated = rotated_cylinder_points[1, :].reshape(y_grid.shape)
        z_rotated = rotated_cylinder_points[2, :].reshape(z_grid.shape)

        # Translate the cylinder to the relative position
        x_translated = x_rotated + relative_position[0]
        y_translated = y_rotated + relative_position[1]
        z_translated = z_rotated + relative_position[2]

        # Plot the cylinder
        ax.plot_surface(x_translated, y_translated, z_translated, color='orange', alpha=0.5, label='Collision Cylinder')
        

        # Add labels and legend
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title(f"Conjunction Analysis for Object {violating_id}")
        plt.show()

        # 2D visualization in relative frame
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ellipse = Ellipse((relative_position[0], relative_position[2]),
                          width=6 * np.sqrt(combined_covariance[0, 0]),
                          height=6 * np.sqrt(combined_covariance[2, 2]),
                          edgecolor='blue', facecolor='cyan', alpha=0.5)
        ax2.add_patch(ellipse)
        ax2.scatter(0, 0, color='red', label='Primary Object')
        ax2.scatter(relative_position[0], relative_position[2], color='blue', label='Secondary Object')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title(f"2D Conjunction Visualization for Object {violating_id}")
        ax2.legend()
        ax2.axis('equal')
    

        # Save plots
        plt.savefig(f"conjunction_3d_{violating_id}.png")
        plt.savefig(f"conjunction_2d_{violating_id}.png")
        plt.show()
        plt.close(fig)
        plt.close(fig2)


    print("Processing and visualization completed.")


def population_analysis( complete_rso , apogee_rso , time_rso , ID):

    a = []
    e = []
    a_ap = []
    e_ap = []
    a_time = []
    e_time = []

    IDs_total = complete_rso.keys()
    IDs_apogee = apogee_rso.keys()
    IDs_time = time_rso.keys()
   
    state_ref = complete_rso[ID]['state']


    kepler_ref = astro.element_conversion.cartesian_to_keplerian(state_ref, 3.986004415e14)

    a.append(kepler_ref[0])
    e.append(kepler_ref[1])
    
    # Cycle for total
    for id in IDs_total:
        if id !=ID:
            state_i = complete_rso[id]['state']

            kepler_i = astro.element_conversion.cartesian_to_keplerian(state_i, 3.986004415e14)

            a.append(kepler_i[0])
            e.append(kepler_i[1])

    for id in IDs_time:
        if id !=ID:
            state_i = time_rso[id]['state']

            kepler_i = astro.element_conversion.cartesian_to_keplerian(state_i, 3.986004415e14)

            a_time.append(kepler_i[0])
            e_time.append(kepler_i[1])

    for id in IDs_apogee:
        if id !=ID:
            state_i = apogee_rso[id]['state']

            kepler_i = astro.element_conversion.cartesian_to_keplerian(state_i, 3.986004415e14)

            a_ap.append(kepler_i[0])
            e_ap.append(kepler_i[1])
    #### Now we plot
    plot_population_analysis(a , e , a_ap , e_ap , a_time , e_time , kepler_ref)

    return
def plot_population_analysis(a, e, a_ap, e_ap, a_time, e_time, kepler_ref):
    plt.figure(figsize=(10, 7))

    # Plot the complete catalog
    plt.scatter(a, e, label='Flitered by apogee-perigee filter', alpha=1)

    #Plot the apogee catalog
    plt.scatter(a_ap, e_ap, label='Remaining Objects', alpha=1)

    # # Plot the time catalog
    #plt.scatter(a_time, e_time)# label='Remainig catalog', alpha=1)

    # Plot the reference element with a clear marker
    plt.scatter(kepler_ref[0], kepler_ref[1], color='red', label='NORAD ID = 40697', marker='X', s=50)


    # Add labels and legend with increased font size
    plt.xlabel('Semi-major Axis (m)', fontsize=16 , fontweight = 'bold')
    plt.ylabel('Eccentricity (-)', fontsize=16 , fontweight = 'bold')
    plt.yscale('log')
    plt.xscale('log')

    # Customizing the legend
    plt.legend(fontsize=15)

    # Adding a black box around the plot
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().spines['left'].set_linewidth(2.5)

    # Increase tick label size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.show()


def population_close_approach(complete_rso , TCA_values, ID):
    a = []
    e = []
    
    IDs_total = complete_rso.keys()
    
    state_ref = complete_rso[ID]['state']


    kepler_ref = astro.element_conversion.cartesian_to_keplerian(state_ref, 3.986004415e14)

    # Cycle for total
    for id in IDs_total:
        if id !=ID:
            state_i = complete_rso[id]['state']

            kepler_i = astro.element_conversion.cartesian_to_keplerian(state_i, 3.986004415e14)

            a.append(kepler_i[0])
            e.append(kepler_i[1])

    plot_population_analysis_TCA(a ,e , TCA_values , kepler_ref)

    return

def plot_population_analysis_TCA(a, e, TCA, kepler_ref):
    plt.figure(figsize=(10, 7))
 # Plot the complete catalog with colormap for TCA distance
    scatter = plt.scatter(a, e, c=TCA, cmap='viridis', alpha=1, label='Filtered by apogee-perigee filter')

    # Plot the reference element with a clear marker
    plt.scatter(kepler_ref[0], kepler_ref[1], color='red', label='NORAD ID = 40697', marker='X', s=100)

    # Add colorbar for TCA distance
    cbar = plt.colorbar(scatter)
    cbar.set_label('TCA Distance (km)', fontsize=16, fontweight='bold')

    # Add labels and legend with increased font size
    plt.xlabel('Semi-major Axis (m)', fontsize=16, fontweight='bold')
    plt.ylabel('Eccentricity (-)', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.xscale('log')

    # Customizing the legend
    plt.legend(fontsize=15)

    # Adding a black box around the plot
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().spines['left'].set_linewidth(2.5)

    # Increase tick label size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.show()

def extract_second_column(file_path):
    second_column = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            if len(values) >= 2:  # Ensure the line has at least two elements
                second_column.append(float(values[1]))  # Convert to float for numerical use
    return second_column


def full_catalog_analysis(rso_catalog):
    LEO_id = []
    MEO_id = []
    GEO_id = []
    HEO_id = []

    IDs_total = rso_catalog.keys()

    for id in IDs_total:
        state = rso_catalog[id]['state']
        kepler = astro.element_conversion.cartesian_to_keplerian(state, 3.986004415e14)

        a = kepler[0] / 1000  # Convert from meters to kilometers
        e = kepler[1]

        # Classifying based on semi-major axis and eccentricity
        if e >= 0.25:
            HEO_id.append(id)
        elif a < 8378:
            LEO_id.append(id)
        elif 8378 <= a < 35786:
            MEO_id.append(id)
        elif 42164 - 200 <= a <= 42164 + 200 and e < 0.1:
            GEO_id.append(id)
    
    return LEO_id, MEO_id, GEO_id, HEO_id

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import matplotlib.image as mpimg

def plot_3D_orbits(rso_catalog, ID, result):
    

    close_time_TDB = []
    Xf_1 = []
    Xf_2 = []
    obj_Id = []

    for result_dict in result:
        for obj_id, metrics in result_dict.items():
            obj_Id.append(obj_id)
            for key, value in metrics.items():
                if key == "close_time_TDB":
                    close_time_TDB.append(value)
                elif key == "X1":
                    Xf_1.append(value)
                elif key == "X2":
                    Xf_2.append(value)

    int_params = dict(
        tudat_integrator='rkf78',
        step=-10,
        max_step=1000,
        min_step=1e-3,
        rtol=1e-12,
        atol=1e-12
    )

    Cd_1 = rso_catalog[ID]['Cd']
    Cr_1 = rso_catalog[ID]['Cr']
    area_1 = rso_catalog[ID]['area']
    mass_1 = rso_catalog[ID]['mass']

    sph_deg = 8
    sph_ord = 8
    central_bodies = ['Earth']
    bodies_to_create = ['Earth', 'Sun', 'Moon']

    state_params_ref = dict(
        central_bodies=central_bodies,
        bodies_to_create=bodies_to_create,
        mass=mass_1,
        area=area_1,
        Cd=Cd_1,
        Cr=Cr_1,
        sph_deg=sph_deg,
        sph_ord=sph_ord
    )

    for id in obj_Id:
        # Find the index and make sure it is an integer
        i = np.where(obj_Id == id)[0][0]
        
        # Convert the result to a NumPy array to ensure numerical operations work
        TCA_epochi = np.array(close_time_TDB[i])
        print(TCA_epochi)
        
        # Subtract 1 hour (3600 seconds) from the TCA epoch
        TCA_epochi_fin = TCA_epochi - 0.8* 3600
        
        # Convert the state values to NumPy arrays
        state_2 = np.array(Xf_2[i])
        state_2 = np.squeeze(state_2)
        state_ref = np.array(Xf_1[i])
        state_ref = np.squeeze(state_ref)
        Cd_2 = rso_catalog[id]['Cd']
        Cr_2 = rso_catalog[id]['Cr']
        area_2 = rso_catalog[id]['area']
        mass_2 = rso_catalog[id]['mass']

        state_params_2 = dict(
            central_bodies=central_bodies,
            bodies_to_create=bodies_to_create,
            mass=mass_2,
            area=area_2,
            Cd=Cd_2,
            Cr=Cr_2,
            sph_deg=sph_deg,
            sph_ord=sph_ord
        )
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

         # Plot the Earth with improved texture
        plot_simple_earth(ax)

        # Set equal scaling
        ax.set_xlim([-7e6, 7e6])
        ax.set_ylim([-7e6, 7e6])
        ax.set_zlim([-7e6, 7e6])

        # Set the font size and boldness
        ax.set_xlabel('X [m]', fontsize=16, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=16, fontweight='bold')
        ax.set_zlabel('Z [m]', fontsize=16, fontweight='bold')

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        ax.xaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.zaxis.get_offset_text().set_fontsize(14)

    

        # Add a black box around the 3D plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Propagate and plot the reference orbit
        _, X_hist_ref = TudatPropagator.propagate_orbit(state_ref, [TCA_epochi, TCA_epochi_fin], state_params_ref, int_params)
        X_ref = X_hist_ref
        ax.plot(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], label=f'Orbit {int(ID)}', linewidth=3)

        _, X_hist_2 = TudatPropagator.propagate_orbit(state_2, [TCA_epochi, TCA_epochi_fin], state_params_2, int_params)
        X_2 = X_hist_2
        ax.plot(X_2[:, 0], X_2[:, 1], X_2[:, 2], label=f'Orbit {int(id)}', linewidth=3)

        ax.scatter(state_ref[0], state_ref[1], state_ref[2], s=30, c='r', label="HIE")

        ax.legend(fontsize=14)
        plt.show()
    return



def plot_simple_earth(ax, radius=3000.1e3, color='b'):
    """
    Plot a simple 3D Earth representation as a sphere.
    
    Parameters:
    - ax: The matplotlib 3D axis object.
    - radius: The radius of the Earth (in km).
    - color: The color of the Earth.
    """
    # Generate sphere coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the Earth as a sphere
    ax.plot_surface(x, y, z, color=color, alpha=0.7, edgecolor='k')

    # Set axis properties for a spherical appearance
    ax.set_box_aspect([1, 1, 1])  # Equal scaling
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

