""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import numpy as np
from tudatpy import constants, numerical_simulation
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.data import save2txt
from tudatpy import astro
from tudatpy.interface import spice
from tudatpy.numerical_simulation import (
    environment,
    environment_setup,
    estimation_setup,
    propagation,
    propagation_setup,
)


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
    mu_earth: float,
    initial_position: float,
    final_position: float, 
    time_of_flight: float,
    departure_epoch: float, 
) -> environment.Ephemeris:
    """
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body.

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """
    print(initial_position)
    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_position,
        final_position,
        time_of_flight,
        mu_earth,
    )


    # Get the departure velocity (3D vector)
    departure_velocity = lambertTargeter.get_departure_velocity()

    # Combine the initial position and departure velocity into a single 6D state vector
    lambert_arc_initial_state = np.concatenate((initial_position, departure_velocity))
    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, mu_earth
    )

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            mu_earth,
        ),
        "",  # for keplerian ephemeris, this argument does not have an effect
    )

    return kepler_ephemeris


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_arc_history(
    lambert_arc_ephemeris: environment.Ephemeris,
    simulation_result: dict
) -> tuple:
    """
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states

    Parameters
    ----------
    lambert_arc_ephemeris : environment.Ephemeris
        Ephemeris object from which the states are to be extracted

    simulation_result : dict
        Dictionary of (numerically propagated) states, from which the keys
        are used to determine the times at which this function is to extract states
        from the lambert arc

    Return
    ------
    Dictionary of Cartesian states of the lambert arc, with the keys (epochs) being the same as those of the input
    simulation_result, the corresponding Cartesian states of the Lambert arc and gravitational acceleration vectors.
    Additionally, returns the epoch at which the spacecraft reaches Earth's Sphere of Influence (SOI) boundary.
    """
    lambert_arc_states = dict()

    for epoch in simulation_result:
        lambert_arc_states[epoch] = lambert_arc_ephemeris.cartesian_state(epoch)

    return lambert_arc_states



# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
) -> numerical_simulation.SingleArcSimulator:
    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : bool
        Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
        is propagated

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """

    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    propagator_settings = get_perturbed_propagator_settings(
        bodies, lambert_arc_initial_state, initial_time, termination_condition
    )

    # Propagate dynamics with required settings
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    return dynamics_simulator


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
) -> numerical_simulation.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
    )

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies)

    # Propagate variational equations
    variational_equations_solver = (
        numerical_simulation.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )
    )

    return variational_equations_solver


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
) -> numerical_simulation.estimation.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : propagation_setup.propagator.PropagatorSettings
        Settings used for the propagation of the dynamics

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )

    return estimation_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )




# STUDENT CODE TASK - full function (except signature and return)
def get_perturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """

    # Create propagation settings.
    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ["RSO"]
    central_bodies = ["Earth"]



    # Define accelerations acting on vehicle.
    acceleration_settings_on_spacecraft = dict( 
        Sun = 
         [
             propagation_setup.acceleration.radiation_pressure(),
             propagation_setup.acceleration.point_mass_gravity()
         ],
        Earth = 
        [
            #propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
            propagation_setup.acceleration.aerodynamic()
        ],
         Moon = 
         [
             propagation_setup.acceleration.point_mass_gravity()
        ],
    )
    
    # Create global accelerations dictionary.
    acceleration_settings = {"RSO": acceleration_settings_on_spacecraft}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )


    # Define required outputs
    dependent_variables_to_save =  [
        propagation_setup.dependent_variable.keplerian_state( "RSO", "Earth" ),
     ]

    coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78
    order_to_use = propagation_setup.integrator.OrderToIntegrate.higher

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        0.01, coefficient_set, order_to_use
    )

    # # Define integrator step settings
    # initial_time_step = 10.0
    # minimum_step_size = 1.0e-12
    # maximum_step_size = np.inf

    # # Retrieve coefficient set
    # coefficient_set = propagation_setup.integrator.rkf_78
    # # Create integrator settings
    # step_size_control_settings = (
    # # propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(current_tolerance, absolute_tolerances[i])
    # # )
    # propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance([[0,0,3,1],[3,0,3,1]],1e-12, 1e-12))
    # step_size_validation_settings = propagation_setup.integrator.step_size_validation(
    # minimum_step_size,
    # maximum_step_size,
    # )

    # Create variable step-size integrator settings
    # integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
    #     initial_time_step,
    #     coefficient_set,
    #     step_size_control_settings,
    #     step_size_validation_settings
    # )


    propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    initial_time,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save,
    )

    return propagator_settings


def tudat_initialize_bodies(bodies_to_create=[]):
    '''
    This function initializes the bodies object for use in the Tudat 
    propagator. For the cases considered, only Earth, Sun, and Moon are needed,
    with Earth as the frame origin.
    
    Parameters
    ------
    bodies_to_create : list, optional (default=[])
        list of bodies to create, if empty, will use default Earth, Sun, Moon
    
    Returns
    ------
    bodies : tudat object
    
    '''

    # Define string names for bodies to be created from default.
    if len(bodies_to_create) == 0:
        bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)
    
    body_settings.add_empty_settings("RSO")

    mass = 200
    Cd = 2.2
    Cr = 1.2
    area = 1

    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [Cd, 0.0, 0.0]
            )

    
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, Cr, occulting_bodies_dict )
            

    # Add the aerodynamic interface to the body settings
    body_settings.get("RSO").aerodynamic_coefficient_settings = aero_coefficient_settings

    
    # Add the aerodynamic interface to the body settings
    body_settings.get("RSO").radiation_pressure_target_settings = radiation_pressure_settings

    bodies = environment_setup.create_system_of_bodies(body_settings)


    bodies.get("RSO").mass = mass #kg
    
    return bodies


def obj_fun(rg1, ra1, dec1, rg2, ra2, dec2, rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs):


        # Initial and final position vectors (ECI)
        r1_eci = position_vector(rg1, ra1, dec1, rgs, rotation_matrix_t1)
        r2_eci = position_vector(rg2, ra2, dec2, rgs, rotation_matrix_t2)


        # ECI position vectors
        # r1_eci = r1_ecef @ rotation_matrix_t1  # Use matrix multiplication (right multiplication)
        # r2_eci = r2_ecef @ rotation_matrix_t2  # Use matrix multiplication (right multiplication) 

        v0_correction = np.zeros(3)
        delta_r_f_norm = np.inf
        prev_delta_r_f_norm = np.inf

        # Convergence threshold
        threshold = 0.001
        initial_state_correction = np.array([0, 0, 0, 0, 0, 0])

        # Iteration counter (to avoid infinite loops)
        max_iterations = 100  # Prevent infinite loops
        iteration = 0

        termination_condition = propagation_setup.propagator.time_termination(
        epochs_tdb_et[1], terminate_exactly_on_final_condition=True
        )

        lambert_arc_ephemeris = get_lambert_problem_result(mu_earth, r1_eci, r2_eci, tof, epochs_tdb_et[0])
        while delta_r_f_norm > threshold and iteration < max_iterations:

                iteration += 1

                # Solve for state transition matrix on current arc
                variational_equations_solver = propagate_variational_equations(
                    epochs_tdb_et[0],
                    termination_condition,
                    bodies,
                    lambert_arc_ephemeris,
                    initial_state_correction = initial_state_correction
                )
                
                state_transition_matrix_history = (
                    variational_equations_solver.state_transition_matrix_history
                )
                state_history = variational_equations_solver.state_history
                lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

                # Get final state transition matrix (and its inverse)
                final_epoch = list(state_transition_matrix_history.keys())[-1]
                initial_epoch = list(state_transition_matrix_history.keys())[1]
                final_state_transition_matrix = state_transition_matrix_history[final_epoch]

                kepler = astro.element_conversion.cartesian_to_keplerian(state_history[initial_epoch], mu_earth)

                print(kepler[0])
                print(kepler[1])

                # Retrieve final state deviation
                final_state_deviation = (
                    state_history[final_epoch] - lambert_history[final_epoch]
                )

                print(f"Final state deviation {final_state_deviation}")
                print(f"Time of Flight {tof}")

                

                # Compute required velocity change at beginning of arc to meet required final state
                # Ensure final_state_transition_matrix is a NumPy array
                final_state_transition_matrix = np.array(final_state_transition_matrix)

                # Ensure final_state_deviation is a NumPy array
                final_state_deviation = np.array(final_state_deviation)

                # Construct the deviation vector (assuming 6D state [position, velocity])
                delta_r_f = final_state_deviation[:3]
                delta_r_f_norm = np.linalg.norm(delta_r_f)


                # Extract relevant submatrices
                Phi_rv = final_state_transition_matrix[:3, 3:]  # (3x3) Position-to-Velocity transition
                Phi_vv = final_state_transition_matrix[3:, 3:]  # (3x3) Velocity-to-Velocity transition

                # Solve for Δv(t_0): Δr(t_1) = Phi_rv * Δv(t_0)
                if np.linalg.cond(Phi_rv) < 1e10:  # Check if invertible
                    delta_v0 = np.linalg.inv(Phi_rv) @ -delta_r_f  # Direct inverse
                    print(f"ΔV₀: {np.linalg.norm(delta_v0)}")
                else:
                    print("Warning: Phi_rv is nearly singular, using pseudo-inverse.")
                    delta_v0 = np.linalg.pinv(Phi_rv) @ -delta_r_f  # Moore-Penrose pseudo-inverse
                    print(f"ΔV₀: {np.linalg.norm(delta_v0)} ")

                # Compute Δv(t_1): Δv(t_1) = Phi_vv * Δv(t_0)
                delta_v1 = Phi_vv @ delta_v0
                

                # 🔹 Print progress
                print(f"Iteration {iteration}: ||Δr_f|| = {delta_r_f_norm:.6f}")
                print(f"ΔV1: {delta_v1}")

                v0_correction = np.array(v0_correction) + np.array(delta_v0)

                initial_state_correction = np.hstack(([0, 0, 0], v0_correction))
                

                # Break if improvement is too small to avoid infinite looping
                if iteration > 1 and abs(prev_delta_r_f_norm - delta_r_f_norm) < 1e-30:
                    print("Convergence reached. Stopping iterations.")
                    break

                prev_delta_r_f_norm = delta_r_f_norm  # Update for next iteration

        # Final Output
        print(f"Final iteration: {iteration}, Final ||Δr_f||: {delta_r_f_norm:.6f}")
        print("Final Initial State Correction:", np.hstack(([0, 0, 0], v0_correction - delta_v0)))

        # Propagate with correction to initial state (use propagate_trajectory function),
        # and its optional initial_state_correction input
        dynamics_simulator = propagate_trajectory(
            epochs_tdb_et[0],
            termination_condition,
            bodies,
            lambert_arc_ephemeris,
            initial_state_correction = np.hstack(([0, 0, 0], [0, 0 ,0])) #v0_correction - delta_v0))
     )

        corr_state_history = dynamics_simulator.propagation_results.state_history
        
        initial_epoch = list(corr_state_history.keys())[0]
        final_epoch = list(corr_state_history.keys())[-1]

        return corr_state_history[initial_epoch], corr_state_history[final_epoch]  

    


def position_vector(rg, ra, dec, r_gs , R1):
    """
    Calculate the position vector in ECEF coordinates from range, elevation, and right ascension.
    """
    pos = rg * np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    r_gs_eci = R1 @ r_gs

    pos[0] += r_gs_eci[0]
    pos[1] += r_gs_eci[1]
    pos[2] += r_gs_eci[2]
    
    return pos



def compute_jacobian(x, rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs, h_rg=1.0, h_angle=0.005):
    """
    Compute the Jacobian matrix for a given state vector using central finite differences.

    Parameters:
    - sv: state vector (sv_1 or sv_2)
    - x: measurement vector (contains rg1, ra1, dec1, rg2, ra2, dec2)
    - h_rg: Perturbation for range (default 1 meter)
    - h_angle: Perturbation for angles (default 0.01 radians)

    Returns:
    - jacobian: Jacobian matrix
    """
    n = len(x)  # Number of measurements
    jacobian_1 = np.zeros((6, 6))  # Jacobian matrix (sv size x x size)
    jacobian_2 = np.zeros((6, 6))

    for j in range(n):
        h = np.zeros(6)
        x_plus = np.copy(x)
        x_minus = np.copy(x)

        if j == 1 or j == 4:
            # Perturb the range measurement by h
            h[j] = h_rg
        else:  # Next three elements are angles (ra1, dec1, ra2, dec2)
            # Perturb the angle measurement by h_angle
            h[j] = h_angle

        # Evaluate the function at different perturbed points
        val = x - 4 * h
        f11, f12 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        val = x - 3 * h
        f21, f22 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        val = x - 2 * h
        f31, f32 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        val = x + 2 * h
        f41, f42 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        val = x + h
        f51, f52 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        val = x + 4 * h
        f61, f62 = obj_fun(val[0], val[1], val[2], val[3], val[4], val[5], rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, rgs)

        # Assign the differences to the Jacobian matrix (treating f11, f12 as row vectors)
        jacobian_1[:, j] = (-1 / 280) * f11 + (4 / 105) * f21 - (1 / 5) * f31 + (1 / 5) * f41 - (4 / 105) * f51 + (1 / 280) * f61
        jacobian_2[:, j] = (-1 / 280) * f12 + (4 / 105) * f22 - (1 / 5) * f32 + (1 / 5) * f42 - (4 / 105) * f52 + (1 / 280) * f62

    return jacobian_1, jacobian_2
