import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def derive_policy(V, problem, reward, gamma):
    Ts = problem["Ts"]  # Transition matrices for each action
    sdim, adim = Ts[0].shape[-1], len(Ts)

    policy = np.zeros(sdim, dtype=int)
    for s in range(sdim):
        # Initialize a list to store the values of each action
        action_values = []
        for a in range(adim):
            # Calculate the value for this action
            # R(x, u) + gamma * sum(p(x'|x,u) * V^*(x'))
            # Here V^*(x') is the optimal value function
            action_value = reward[s, a] + gamma * np.dot(Ts[a][s], V)
            action_values.append(action_value)

        # Select the action which gives the maximum value
        # Taking the argmax of the bellman equation with optimal value function
        best_action = np.argmax(action_values)
        policy[s] = best_action

    return policy


def simulate_mdp(policy, problem, start_state, N=100):
    n = problem["n"]
    Ts = problem["Ts"]  # Not each row sums to 1 because of floating point precision
    states = [start_state]
    current_state = start_state

    for _ in range(N):
        # Take the optimal action given current state
        action = policy[current_state]
        # Get the transition probabilities for the chosen action
        probabilities = Ts[action][current_state].numpy()
        # NOTE: I got error on the np.random.choice call below saying that probs did not sum to 1
        # I tried a lot of things to solve it and the error was because of floating point precision
        # So I just normalized the probabilities and it did not solve
        # But turning tf tensor to numpy array just solved it magically
        # Numpy handled floating point precision much better

        # Generate the next state based on the transition probabilities for the chosen action
        next_state = np.random.choice(np.arange(n * n), p=probabilities)
        states.append(next_state)
        current_state = next_state

    # Convert state indices back to (x, y) coordinates for visualization
    coords = [(state % n, state // n) for state in states]
    return coords



def visualize_policy(policy_grid, path_coords):

    # Create a color map for different actions
    cmap = ListedColormap(["red", "blue", "green", "purple"])  # Assuming 4 actions
    plt.figure(figsize=(10, 8))
    plt.imshow(policy_grid, cmap=cmap, origin="lower")
    plt.colorbar(
        ticks=[0, 1, 2, 3], label="Actions (0: right, 1: up, 2: left, 3: down)"
    )
    plt.title("Drone Navigation Policy Heatmap")

    # Overlay the path
    xs, ys = zip(*path_coords)
    plt.plot(xs, ys, color="gold", marker="o", label="Drone Path")
    plt.legend()

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(False)
    plt.show()


