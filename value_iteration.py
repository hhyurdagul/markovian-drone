import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import generate_problem, visualize_value_function
from utils_other import derive_policy, simulate_mdp, visualize_policy

def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]  # Transition matrices for each action
    sdim, adim = Ts[0].shape[-1], len(Ts)  # State and action dimension
    # sdim = 400, adim = 4
    V = tf.zeros([sdim])  # Initial value function estimate

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for _ in range(10000):
        ######### Your code starts here #########

        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        V_prev = tf.identity(V)
        V_new = tf.zeros_like(V)

        # Ts is a 4 element python list of transition matrices for 4 actions
        for action in range(adim):
            # Transition probabilies -> p(x'|x,u)
            T = Ts[action]

            # R(x, u) + gamma * sum(p(x'|x,u) * V(x')))
            V_action = tf.reduce_sum(T * V_prev, axis=1)
            V_temp = reward[:, action] + gam * V_action

            # V changes all the time and takes the best action
            V_new = tf.maximum(V_new, V_temp)

        # Apply the terminal state mask: Set the value of terminal states to 0
        V_new = tf.where(terminal_mask, reward[:, 0], V_new)
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        err = tf.linalg.norm(V_new - V_prev)

        ######### Your code ends here ###########

        V = V_new

        if err < 1e-7:
            break

    return V


# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, _ = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.bool)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("Value iteration")
    plt.show()

    # derive the policy
    policy = derive_policy(V_opt, problem, reward, gam)

    # simulate the policy
    start_state = problem["pos2idx"][0, 0]
    coords = simulate_mdp(policy, problem, start_state, N=100)

    # Since filled each state with the best action
    # We loop the state list first and then the action list
    # Which means inside lists are first and outside lists are second
    # After all the equations finished we have to transpose the policy
    # To make actions are outer list and states are inner list
    # We also have to change the order of x, y as the same way 

    policy_grid = np.array(policy).reshape(n, n).T
    path_coords = [(c[1], c[0]) for c in coords]

    visualize_policy(policy_grid, path_coords)

if __name__ == "__main__":
    main()
