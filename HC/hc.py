import numpy as np
from tsp_env import TSPEnv
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


def plot_and_save_tour(locations, current_tour, best_tour, filename, current_distance, best_distance, iteration):
    plt.figure(figsize=(6, 5))

    x_current = [locations[i][0] for i in current_tour + [current_tour[0]]]
    y_current = [locations[i][1] for i in current_tour + [current_tour[0]]]
    plt.plot(x_current, y_current, 'o-', color='lightgray', alpha=0.5, label='Current Tour')

    x_best = [locations[i][0] for i in best_tour + [best_tour[0]]]
    y_best = [locations[i][1] for i in best_tour + [best_tour[0]]]
    plt.plot(x_best, y_best, 'o-', color='green', markerfacecolor='red', label='Best Tour')

    for i, node in enumerate(best_tour):
        plt.text(locations[node][0], locations[node][1], str(node), fontsize=7, ha='right')

    plt.title(f"Hill Climbing (Iter {iteration})\nBest Dist: {best_distance:.2f} | Current: {current_distance:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def hill_climbing(env: TSPEnv, max_iterations=1000, frame_dir="temp_plots_hc"):
    state = env.reset()
    locations = state[0, :, :2]
    num_nodes = locations.shape[0]

    current_solution = list(range(num_nodes))
    np.random.shuffle(current_solution)
    current_distance = total_distance(locations, current_solution)

    best_solution = current_solution.copy()
    best_distance = current_distance

    distances = []
    gif_frames = []

    os.makedirs(frame_dir, exist_ok=True)

    for iteration in range(max_iterations):
        improved = False

        for _ in range(100):  # Try 100 random neighbors
            i, j = np.random.choice(num_nodes, 2, replace=False)
            new_solution = current_solution.copy()
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_distance = total_distance(locations, new_solution)

            if new_distance < current_distance:
                current_solution = new_solution
                current_distance = new_distance
                improved = True

                if current_distance < best_distance:
                    best_solution = current_solution.copy()
                    best_distance = current_distance

                break  # Accept the first improving move (steepest ascent variant)

        if not improved:
            break  # No improvement, local maximum reached

        distances.append(best_distance)

        if iteration % 10 == 0 or iteration == max_iterations - 1:
            filename = os.path.join(frame_dir, f"frame_{iteration:04d}.png")
            plot_and_save_tour(locations, current_solution, best_solution, filename, current_distance, best_distance, iteration)
            gif_frames.append(imageio.imread(filename))

    imageio.mimsave("tsp_hc_progress.gif", gif_frames, fps=10)

    return best_solution, best_distance, distances, locations


if __name__ == "__main__":
    num_runs = 5
    run_times = []
    all_best_distances = []

    for run in range(num_runs):
        print(f"\n--- Hill Climbing Run {run + 1} ---")
        #env = TSPEnv(num_nodes=30, batch_size=1, num_draw=1)
        env = TSPEnv()
        env.num_nodes = 30
        env.batch_size = 1
        env.num_draw = 1  # Optional, if your version uses this
        best_tour, best_dist, dist_progression, coords = hill_climbing(env)
        start_time = time.time()
        best_tour, best_dist, dist_progression, coords = hill_climbing(env)
        end_time = time.time()

        run_duration = end_time - start_time
        run_times.append(run_duration)
        all_best_distances.append(best_dist)

        print(f"Best tour: {best_tour}")
        print(f"Best distance: {best_dist:.2f}")
        print(f"Time taken: {run_duration:.2f} seconds")

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_runs + 1), run_times, color='lightgreen')
    plt.axhline(np.mean(run_times), color='red', linestyle='--', label=f'Average: {np.mean(run_times):.2f}s')
    plt.title("Hill Climbing - Time Taken for Each Run")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.xticks(range(1, num_runs + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nGIF saved as tsp_hc_progress.gif")

    # Cleanup
    shutil.rmtree('temp_plots_hc', ignore_errors=True)
