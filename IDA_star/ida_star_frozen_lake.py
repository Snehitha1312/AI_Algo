import time
import os
import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Create the Frozen Lake environment (8x8)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array")
env = env.unwrapped  # Access underlying environment for transition probabilities

def heuristic(state):
    """Manhattan distance heuristic for Frozen Lake"""
    x = state // 8
    y = state % 8
    return (7 - x) + (7 - y)  # Distance to goal (7,7)

def ida_star(env, max_time=600):
    start_time = time.time()
    env.reset()
    goal_state = env.observation_space.n - 1
    
    search_history = []  # Stores all visited states (for GIF)

    def search(path, g, limit, start_time):
        current_state = path[-1]
        f = g + heuristic(current_state)
        
        # Track every move the agent makes
        search_history.append(current_state)

        # Timeout check
        if time.time() - start_time > max_time:
            return float('inf'), None
        
        if f > limit:
            return f, None  # Return exceeded value
            
        if current_state == goal_state:
            return f, path  # Return solution
            
        min_exceed = float('inf')
        for action in range(env.action_space.n):
            if current_state in env.P:
                for prob, next_state, reward, done in env.P[current_state][action]:
                    if next_state not in path:  # Prevent cycles
                        # Recursive search with updated path and cost
                        new_g = g + 1
                        res, found_path = search(
                            path + [next_state], 
                            new_g, 
                            limit, 
                            start_time
                        )
                        
                        if found_path is not None:
                            return res, found_path
                        if res < min_exceed:
                            min_exceed = res
        return min_exceed, None

    # Initial setup
    start_state = 0
    limit = heuristic(start_state)
    path = [start_state]
    
    while time.time() - start_time < max_time:
        res, found_path = search(path, 0, limit, start_time)
        
        if found_path is not None:
            execution_time = time.time() - start_time
            return found_path, len(found_path)-1, execution_time, True, search_history
        
        if res == float('inf'):
            break  # No solution exists
            
        limit = res  # Update threshold for next iteration

    execution_time = time.time() - start_time
    return None, float('inf'), execution_time, False, search_history

# Function to create and save a GIF of the best path only
def generate_best_path_gif(env, best_path_states, filename="ida_star_best_path.gif"):
    frames = []
    env.reset()

    for state in best_path_states:
        env.s = state  # Set the environment to the current state
        frames.append(env.render())  # Capture frame

    env.close()

    # Save GIF in the current directory
    save_path = os.path.join(os.getcwd(), filename)
    imageio.mimsave(save_path, frames, duration=0.5)  # Slower playback for clarity
    print(f"GIF saved at: {save_path}")
    return save_path

# Run IDA* multiple times (same structure as BnB)
num_runs = 5
ida_execution_times = []
ida_successful_runs = 0
ida_best_path = None
ida_best_cost = float('inf')

for i in range(num_runs):
    print(f"IDA* Run {i+1}:")
    path, cost, exec_time, reached_goal, visited_states = ida_star(env)
    
    if reached_goal:
        ida_successful_runs += 1
        ida_execution_times.append(exec_time)
        print(f"  Best Path: {path}")
        print(f"  Best Cost: {cost}")
        print(f"  Execution Time: {exec_time:.4f} seconds\n")
        
        if cost < ida_best_cost:
            ida_best_cost = cost
            ida_best_path = path

# Generate IDA* GIF (best path only)
if ida_best_path:
    gif_path = generate_best_path_gif(env, ida_best_path)

# Compute average execution time
if ida_successful_runs > 0:
    avg_time = np.mean(ida_execution_times)
    print(f"Average Execution Time: {avg_time:.4f} seconds")
else:
    print("No successful runs, unable to compute average execution time.")

# Plotting execution times
if ida_successful_runs > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, ida_successful_runs + 1), ida_execution_times, marker='o', linestyle='-', color='r')
    plt.xlabel("Run Number")
    plt.ylabel("Execution Time (seconds)")
    plt.title("IDA* Execution Time over Multiple Runs (8x8 Frozen Lake)")
    plt.grid(True)
    plt.show()
else:
    print("No successful runs to plot.")
