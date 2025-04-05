import numpy as np
from tsp_env import TSPEnv  # make sure tsp_env.py is in the same directory
import matplotlib.pyplot as plt
import shutil
import imageio
import os
import time


def total_distance(locations, tour):
    dist = 0
    for i in range(len(tour)):
        from_node = locations[tour[i]]
        to_node = locations[tour[(i + 1) % len(tour)]]
        dist += np.linalg.norm(from_node - to_node)
    return dist


def plot_and_save_tour(locations, tour, filename):
    plt.figure(figsize=(6, 5))
    x = [locations[i][0] for i in tour + [tour[0]]]
    y = [locations[i][1] for i in tour + [tour[0]]]
    plt.plot(x, y, 'o-', color='blue', markerfacecolor='red')
    for i, node in enumerate(tour):
        plt.text(locations[node][0], locations[node][1], str(node), fontsize=8, ha='right')
    plt.title("TSP Tour (Progress)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def simulated_annealing(env: TSPEnv, max_iterations=500, initial_temp=1000, cooling_rate=0.995):
    state = env.reset()
    locations = state[0, :, :2]
    num_nodes = locations.shape[0]

    current_solution = list(range(num_nodes))
    np.random.shuffle(current_solution)
    current_distance = total_distance(locations, current_solution)

    best_solution = current_solution.copy()
    best_distance = current_distance

    temp = initial_temp
    distances = []
    gif_frames = []

    for iteration in range(max_iterations):
        i, j = np.random.choice(num_nodes, 2, replace=False)
        new_solution = current_solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_distance = total_distance(locations, new_solution)

        if new_distance < current_distance or np.random.rand() < np.exp(-(new_distance - current_distance) / temp):
            current_solution = new_solution
            current_distance = new_distance

            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance

        temp *= cooling_rate
        distances.append(best_distance)

        if iteration % 20 == 0 or iteration == max_iterations - 1:
            filename = f"frame_{iteration}.png"
            plot_and_save_tour(locations, best_solution, filename)
            gif_frames.append(imageio.imread(filename))

    imageio.mimsave("tsp_sa_progress.gif", gif_frames, fps=5)

    for file in os.listdir():
        if file.startswith("frame_") and file.endswith(".png"):
            os.remove(file)

    return best_solution, best_distance, distances, locations


def visualize_tour(locations, tour, title="TSP Tour"):
    plt.figure(figsize=(8, 6))
    x = [locations[i][0] for i in tour + [tour[0]]]
    y = [locations[i][1] for i in tour + [tour[0]]]
    plt.plot(x, y, 'o-', color='blue', markerfacecolor='red')
    for i, node in enumerate(tour):
        plt.text(locations[node][0], locations[node][1], str(node), fontsize=9, ha='right')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    num_runs = 5
    run_times = []
    all_best_distances = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        env = TSPEnv(num_nodes=20, batch_size=1, num_draw=1)

        start_time = time.time()
        best_tour, best_dist, dist_progression, coords = simulated_annealing(env)
        end_time = time.time()

        run_duration = end_time - start_time
        run_times.append(run_duration)
        all_best_distances.append(best_dist)

        print(f"Best tour: {best_tour}")
        print(f"Best distance: {best_dist:.2f}")
        print(f"Time taken: {run_duration:.2f} seconds")

    # Plot the best distance for each run
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_runs + 1), run_times, color='skyblue')
    plt.axhline(np.mean(run_times), color='red', linestyle='--', label=f'Average: {np.mean(run_times):.2f}s')
    plt.title("Time Taken for Each Run")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.xticks(range(1, num_runs + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nGIF saved as tsp_sa_progress.gif")


    shutil.rmtree('temp_plots', ignore_errors=True)
