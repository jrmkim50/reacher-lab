import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
L1 = 0.08 # length of link 1
L2 = 0.11 # length of link 2

def calculate_forward_kinematics_robot(joint_angles):
    """Calculate xyz coordinates of end-effector given joint angles.

    Use forward kinematics equations to calculate the xyz coordinates of the end-effector
    given some joint angles.

    Args:
      joint_angles: numpy array of 3 elements [TODO names]. Numpy array of 3 elements.
    Returns:
      xyz coordinates of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    theta_1, theta_2, theta_3 = joint_angles
    r_c0_e = np.array([0, 0, L2])
    R_b_c = np.array(
      [
        [math.cos(theta_3), 0, -math.sin(theta_3)],
        [0, 1, 0],
        [math.sin(theta_3), 0, math.cos(theta_3)]
      ]
    )
    R_a_b = np.array(
      [
        [math.cos(theta_2), 0, -math.sin(theta_2)],
        [0, 1, 0],
        [math.sin(theta_2), 0, math.cos(theta_2)]
      ]
    )
    R_n_a = np.array(
      [
        [math.cos(-theta_1), -math.sin(-theta_1), 0],
        [math.sin(-theta_1), math.cos(-theta_1), 0],
        [0, 0, 1]
      ]
    )
    r_b0_e = np.array([0, 0, L1]) + np.dot(R_b_c, r_c0_e)
    r_a0_e = np.array([0, -HIP_OFFSET, 0]) + np.dot(R_a_b, r_b0_e)
    return np.dot(R_n_a, r_a0_e)

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics loss.

    Calculate the Euclidean distance between the desired end-effector position and
    the end-effector position resulting from the given 'guess' joint angles.

    Args:
      end_effector_pos: desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: guess at joint angles to achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      Euclidean distance between end_effector_pos and guess. Returns float.
    """
    # TODO for students: Implement this function. ~1-5 lines of code.
    norm = np.linalg.norm(end_effector_pos - calculate_forward_kinematics_robot(guess))**2
    return 0.5*norm

def calculate_jacobian(joint_angles):
    """Calculate the jacobian of the end-effector position wrt joint angles.
    
    Calculate the jacobian, which is a matrix of the partial derivatives
    of the forward kinematics with respect to the joint angles 
    arranged as such:
    
    dx/dtheta1 dx/dtheta2 dx/dtheta3
    dy/dtheta1 dy/dtheta2 dy/dtheta3
    dz/dtheta1 dz/dtheta2 dz/dtheta3
    
    Args:
      joint_angles: joint angles of robot arm. Numpy array of 3 elements.
    
    Returns:
      Jacobian matrix. Numpy 3x3 array.
    """
    # TODO for students: Implement this function. ~5-10 lines of code.
    jacobian = np.zeros((3, 3))
    delta = 1e-3
    for theta_idx in range(3):
      offset = np.zeros((3))
      offset[theta_idx] = delta
      finite_difference = (calculate_forward_kinematics_robot(joint_angles+offset) - calculate_forward_kinematics_robot(joint_angles)) / delta
      jacobian[:, theta_idx] = finite_difference
    return jacobian

def calculate_inverse_kinematics(end_effector_pos, guess):
    """Calculates joint angles given desired xyz coordinates.

    Use gradient descent to minimize the inverse kinematics loss function. The
    joint angles that minimize the loss function are the joint angles that give 
    the smallest error from the actual resulting end-effector position to the
    desired end-effector position. You should use the jacobain function
    you wrote above.

    Args:
      end_effector_pos: Desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: Guess at joint angles that achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      Joint angles that correspond to given desired end-effector position. Numpy array with 3 elements.
    """
    # TODO for students: Implement this function. ~10-20 lines of code.
    # joint_angles = np.array([0.0, 0.0, 0.0])
    joint_angles = guess
    C_prev = 0
    epsilon = 1e-5
    alpha = 20
    for iter in range(1000):
      print(iter, C_prev)
      jacobian = calculate_jacobian(joint_angles)
      cost_gradient = jacobian.T @ (calculate_forward_kinematics_robot(joint_angles) - end_effector_pos)
      joint_angles -= alpha * cost_gradient
      if ik_cost(end_effector_pos, joint_angles) < epsilon:
        return joint_angles
      else:
        C_prev = ik_cost(end_effector_pos, joint_angles)
    return joint_angles