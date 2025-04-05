import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from tqdm import tqdm
from HC.tsp_env import TSPEnv  # Adjust path as needed


def hill_climbing_tsp(env, timeout=600, render_gif=False, gif_path="hill_climbing.gif"):
    obs = env.reset()
    start_time = time.time()

    total_reward = 0
    frames = []

    current_node = env.depots[0][0]
    unvisited = set(range(env.num_nodes)) - {current_node}
    tour = [current_node]

    while unvisited and (time.time() - start_time) < timeout:
        neighbors = list(unvisited)
        np.random.shuffle(neighbors)

        best_move = None
        best_reward = float('-inf')

        for next_node in neighbors:
            action = np.array([[next_node]])
            _, reward, done, _ = env.step(action)
            if reward[0] > best_reward:
                best_reward = reward[0]
                best_move = next_node

        if best_move is not None:
            tour.append(best_move)
            unvisited.remove(best_move)
            action = np.array([[best_move]])
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]

            if render_gif:
                frame = env.render(mode="rgb_array")
                frames.append(frame)
        else:
            break

    if render_gif and frames:
        imageio.mimsave(gif_path, frames, fps=1)

    time_taken = time.time() - start_time
    return time_taken, total_reward, tour


def run_experiments(num_runs=5, timeout=600):
    times = []
    rewards = []
    best_paths = []

    for i in range(num_runs):
        print(f"--- Run {i+1} ---")
        env = TSPEnv(num_nodes=20, batch_size=1, num_draw=1)
        gif_path = f"gifs/hc_run_{i+1}.gif"
        t, reward, path = hill_climbing_tsp(env, timeout=timeout, render_gif=True, gif_path=gif_path)
        print(f"Time: {t:.2f}s | Reward: {reward:.2f} | Path: {path}")

        times.append(t)
        rewards.append(reward)
        best_paths.append(path)

    avg_time = np.mean(times)

    # Plot timing
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_runs + 1), times, marker='o', label='Run Time')
    plt.axhline(avg_time, color='r', linestyle='--', label=f'Avg Time: {avg_time:.2f}s')
    plt.title("Time to Reach Optimum in Hill Climbing (TSP)")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("hill_climbing_times.png")
    plt.close()

    print(f"\nAverage Time Taken: {avg_time:.2f}s")
    return times, rewards, best_paths


if __name__ == "__main__":
    Path("gifs").mkdir(exist_ok=True)
    run_experiments()
