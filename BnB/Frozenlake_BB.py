import time
import os
import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Create the Frozen Lake environment (8x8)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array")
env = env.unwrapped  # Access underlying environment for transition probabilities

# Branch and Bound using a stack (LIFO)
def branch_and_bound(env, max_time=600):
    start_time = time.time()
    env.reset()

    # Stack for DFS-based search (Branch & Bound)
    stack = [(0, [0], 0)]  # (state, path, cost)

    best_cost = float('inf')
    best_path = None
    reached_goal = False

    while stack:
        current_state, path, cost = stack.pop()

        # Check timeout
        if time.time() - start_time > max_time:
            break

        # Goal check
        if current_state == env.observation_space.n - 1:  # Reached goal state
            if cost < best_cost:
                best_cost = cost
                best_path = path
            reached_goal = True
            continue

        # Expand to valid neighbors using env.unwrapped.P
        for action in range(env.action_space.n):
            if current_state in env.P:
                for prob, next_state, reward, done in env.P[current_state][action]:
                    if next_state not in path:  # Avoid cycles
                        new_cost = cost + 1
                        if new_cost < best_cost:  # Branch and Bound pruning
                            stack.append((next_state, path + [next_state], new_cost))

    execution_time = time.time() - start_time
    return best_path, best_cost, execution_time, reached_goal

# Function to create and save a GIF of the best path execution
def generate_gif(env, path, filename="bnb_frozenlake_8x8.gif"):
    frames = []
    env.reset()

    for state in path:
        env.s = state  # Set the environment to the current state
        frames.append(env.render())  # Capture frame

    env.close()

    # Save GIF in the current directory
    save_path = os.path.join(os.getcwd(), filename)
    imageio.mimsave(save_path, frames, duration=0.5)
    print(f" GIF saved at: {save_path}")
    return save_path

# Run BnB multiple times and collect results
num_runs = 5
execution_times = []
successful_runs = 0
best_overall_path = None
best_overall_cost = float('inf')

for i in range(num_runs):
    print(f"Run {i+1}:")
    path, cost, exec_time, reached_goal = branch_and_bound(env)

    if reached_goal:
        successful_runs += 1
        execution_times.append(exec_time)
        print(f"  Best Path: {path}")
        print(f"  Best Cost: {cost}")
        print(f"  Execution Time: {exec_time:.4f} seconds\n")

        # Track the best path across all runs
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_path = path
    else:
        print("  BnB failed to reach the goal.\n")

# Compute average execution time
if successful_runs > 0:
    avg_time = np.mean(execution_times)
    print(f"Average Execution Time: {avg_time:.4f} seconds")
else:
    print("No successful runs, unable to compute average execution time.")

# Generate GIF for the best path found
if best_overall_path:
    gif_path = generate_gif(env, best_overall_path)

# Plot execution times
plt.figure(figsize=(8, 5))
plt.plot(range(1, successful_runs + 1), execution_times, marker='o', linestyle='--', color='b')
plt.xlabel("Run Number")
plt.ylabel("Execution Time (seconds)")
plt.title("BnB Execution Time over Multiple Runs (8x8)")
plt.grid()
plt.show()

