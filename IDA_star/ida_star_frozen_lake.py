
# import gymnasium as gym
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import imageio  # For GIF creation

# # Initialize Frozen Lake Environment
# env = gym.make("FrozenLake-v1", desc=None, is_slippery=False, render_mode="rgb_array")

# # Function to compute Manhattan Distance as heuristic
# def get_manhattan_distance(state, goal, size):
#     x1, y1 = divmod(state, size)
#     x2, y2 = divmod(goal, size)
#     return abs(x1 - x2) + abs(y1 - y2)

# # Function to get possible moves from current state
# def get_neighbors(state, size):
#     row, col = divmod(state, size)
#     neighbors = []
#     actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

#     for i, (dx, dy) in enumerate(actions):
#         new_row, new_col = row + dx, col + dy
#         if 0 <= new_row < size and 0 <= new_col < size:
#             neighbors.append((i, new_row * size + new_col))  # (Action, New State)
    
#     return neighbors

# # IDA* Algorithm Implementation
# def ida_star(env, start, goal):
#     size = int(np.sqrt(env.observation_space.n))  # Get grid size (e.g., 4x4)
#     path_taken = []  # Store path for visualization

#     def search(node, g, threshold, path):
#         f = g + get_manhattan_distance(node, goal, size)
#         if f > threshold:
#             return f
        
#         if node == goal:
#             return "FOUND"
        
#         min_cost = float("inf")
#         for action, neighbor in get_neighbors(node, size):
#             if neighbor in path:  # Avoid loops
#                 continue
#             path.add(neighbor)
#             path_taken.append((action, neighbor))  # Save (action, state) for visualization
#             temp = search(neighbor, g + 1, threshold, path)
#             if temp == "FOUND":
#                 return "FOUND"
#             if temp < min_cost:
#                 min_cost = temp
#             path.remove(neighbor)
        
#         return min_cost

#     # Iterative deepening loop
#     threshold = get_manhattan_distance(start, goal, size)
#     while True:
#         path = {start}
#         path_taken.clear()
#         path_taken.append((None, start))  # Start with None action
#         result = search(start, 0, threshold, path)
#         if result == "FOUND":
#             return "Path found", path_taken
#         if result == float("inf"):
#             return "No path found", []
#         threshold = result  # Increase threshold

# # Get Start and Goal Positions
# start_state = 0  # 'S' is at index 0 in a 4x4 Frozen Lake
# goal_state = env.observation_space.n - 1  # 'G' is at the last index

# # Run IDA* and Get Path
# result, path_taken = ida_star(env, start_state, goal_state)
# print("Result:", result)

# # Generate and Save GIF with Proper Movements
# frames = []
# env.reset()

# # Reset the environment for visualization
# obs, _ = env.reset()
# frames.append(env.render())  # Capture initial frame

# for action, state in path_taken[1:]:  # Skip the start state
#     obs, _, _, _, _ = env.step(action)  # Step with the chosen action
#     frames.append(env.render())  # Capture frame after action

# # Save the GIF
# imageio.mimsave("ida_star_frozenlake.gif", frames, duration=0.5)
# print("GIF saved as 'ida_star_frozenlake.gif'")

# # Measure Execution Time for Multiple Runs
# runs = 5
# times = []

# for _ in range(runs):
#     start_time = time.perf_counter()
#     result, _ = ida_star(env, start_state, goal_state)
#     end_time = time.perf_counter()
#     times.append(end_time - start_time)

# print("Average Execution Time:", np.mean(times))

# # Plot Execution Time
# plt.plot(range(1, runs + 1), times, marker='o', linestyle='-')
# plt.xlabel("Run Number")
# plt.ylabel("Execution Time (seconds)")
# plt.title("IDA* Execution Time on Frozen Lake")
# plt.show()
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
    
    def search(path, g, limit, start_time):
        current_state = path[-1]
        f = g + heuristic(current_state)
        
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
            return found_path, len(found_path)-1, execution_time, True
            
        if res == float('inf'):
            break  # No solution exists
            
        limit = res  # Update threshold for next iteration

    execution_time = time.time() - start_time
    return None, float('inf'), execution_time, False

# Function to create and save a GIF of the best path execution
def generate_gif(env, path, filename="ida_star_frozenlake_8x8.gif"):
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

# Run IDA* multiple times (same structure as BnB)
num_runs = 5
ida_execution_times = []
ida_successful_runs = 0
ida_best_path = None
ida_best_cost = float('inf')

for i in range(num_runs):
    print(f"IDA* Run {i+1}:")
    path, cost, exec_time, reached_goal = ida_star(env)
    
    if reached_goal:
        ida_successful_runs += 1
        ida_execution_times.append(exec_time)
        print(f"  Best Path: {path}")
        print(f"  Best Cost: {cost}")
        print(f"  Execution Time: {exec_time:.4f} seconds\n")
        
        if cost < ida_best_cost:
            ida_best_cost = cost
            ida_best_path = path
    else:
        print("  IDA* failed to reach the goal.\n")

# Compute average execution time
if ida_successful_runs > 0:
    avg_time = np.mean(ida_execution_times)
    print(f"Average Execution Time: {avg_time:.4f} seconds")
else:
    print("No successful runs, unable to compute average execution time.")

# Generate IDA* GIF
if ida_best_path:
    gif_path = generate_gif(env, ida_best_path)

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

