import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- Configuration ---
CONTAINER_WIDTH = 85.60
CONTAINER_HEIGHT = 54

# Coin specifications: (Name, Diameter, Count)
COIN_SPECS = [
    ("Quarter", 24.26, 3),
    ("Dime",    17.91, 2),
    ("Nickel",  21.21, 1),
    ("Penny",   19.05, 4),
]

# Simulated Annealing Parameters
INITIAL_TEMPERATURE = 20.0
COOLING_RATE = 0.99999 # Slower cooling often better, but takes longer
MAX_ITERATIONS = 1000000 # Increase for potentially better results
MOVE_SCALE_INITIAL = 30.0 # Initial max distance a coin can move in one step
MOVE_SCALE_FINAL = 0.1 # Final max distance
STOPPING_COST = 1e-9 # Stop if cost is effectively zero

# --- Prepare Coin Data ---
coins = []
for name, diameter, count in COIN_SPECS:
    radius = diameter / 2.0
    for _ in range(count):
        coins.append({"name": name, "radius": radius, "id": len(coins)})

num_coins = len(coins)
radii = np.array([c['radius'] for c in coins])

# --- Helper Functions ---

def calculate_cost(positions, radii, container_width, container_height):
    """
    Calculates the cost (badness) of a configuration.
    Cost is based on overlaps between circles and circles extending outside the container.
    Goal is to minimize cost to zero.
    """
    cost = 0.0
    n = len(positions)

    # 1. Boundary Violations
    for i in range(n):
        r = radii[i]
        x, y = positions[i]
        # Left boundary
        cost += max(0, r - x)**2
        # Right boundary
        cost += max(0, (x + r) - container_width)**2
        # Bottom boundary
        cost += max(0, r - y)**2
        # Top boundary
        cost += max(0, (y + r) - container_height)**2

    # 2. Overlap Violations
    for i in range(n):
        for j in range(i + 1, n):
            pos_i = positions[i]
            pos_j = positions[j]
            r_i = radii[i]
            r_j = radii[j]

            dist_sq = np.sum((pos_i - pos_j)**2)
            min_dist = r_i + r_j
            min_dist_sq = min_dist**2

            if dist_sq < min_dist_sq:
                # Calculate overlap distance
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
                overlap = min_dist - dist
                cost += overlap**2 # Penalize quadratically

    return cost

def get_random_initial_state(num_coins, radii, container_width, container_height):
    """Creates a random starting arrangement."""
    positions = np.zeros((num_coins, 2))
    for i in range(num_coins):
        r = radii[i]
        positions[i, 0] = random.uniform(r, container_width - r)
        positions[i, 1] = random.uniform(r, container_height - r)
    return positions

def plot_arrangement(positions, radii, container_width, container_height, title="Coin Packing Arrangement"):
    """Visualizes the arrangement."""
    fig, ax = plt.subplots(1, figsize=(8.56, 5.4)) # Scale fig size roughly
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_aspect('equal', adjustable='box')

    # Draw container
    rect = patches.Rectangle((0, 0), container_width, container_height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw circles
    for i in range(len(positions)):
        circle = patches.Circle(positions[i], radii[i], fill=True, alpha=0.6, edgecolor='black')
        ax.add_patch(circle)
        # Optional: Label coins
        # ax.text(positions[i, 0], positions[i, 1], coins[i]['id'], ha='center', va='center', fontsize=8)

    plt.title(title)
    plt.xlabel("Width (mm)")
    plt.ylabel("Height (mm)")
    plt.gca().invert_yaxis() # Put (0,0) at top-left if desired, comment out for bottom-left
    plt.show()

# --- Simulated Annealing Main Logic ---

def solve_packing():
    print("Attempting to pack coins using Simulated Annealing...")
    start_time = time.time()

    current_positions = get_random_initial_state(num_coins, radii, CONTAINER_WIDTH, CONTAINER_HEIGHT)
    current_cost = calculate_cost(current_positions, radii, CONTAINER_WIDTH, CONTAINER_HEIGHT)

    best_positions = np.copy(current_positions)
    best_cost = current_cost

    temperature = INITIAL_TEMPERATURE

    print(f"Initial Cost: {current_cost:.4f}")

    for iteration in range(MAX_ITERATIONS):
        if current_cost < STOPPING_COST:
            print(f"\nSolution found at iteration {iteration}!")
            best_positions = np.copy(current_positions)
            best_cost = current_cost
            break

        # Generate neighbor state: move one coin randomly
        new_positions = np.copy(current_positions)
        coin_to_move = random.randrange(num_coins)
        r = radii[coin_to_move]

        # Adjust move scale based on temperature/iteration
        # Linear interpolation from initial to final scale
        progress = iteration / MAX_ITERATIONS
        current_move_scale = MOVE_SCALE_INITIAL * (1 - progress) + MOVE_SCALE_FINAL * progress
        # Alternative: scale with temperature: current_move_scale = max(MOVE_SCALE_FINAL, temperature * 0.1)


        # Generate random move vector
        move_x = random.uniform(-current_move_scale, current_move_scale)
        move_y = random.uniform(-current_move_scale, current_move_scale)

        # Apply move, ensuring center stays roughly within bounds initially
        new_positions[coin_to_move, 0] = np.clip(new_positions[coin_to_move, 0] + move_x, 0, CONTAINER_WIDTH)
        new_positions[coin_to_move, 1] = np.clip(new_positions[coin_to_move, 1] + move_y, 0, CONTAINER_HEIGHT)


        new_cost = calculate_cost(new_positions, radii, CONTAINER_WIDTH, CONTAINER_HEIGHT)

        # Metropolis acceptance criterion
        cost_diff = new_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_positions = new_positions
            current_cost = new_cost

            # Update best found solution
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = np.copy(current_positions)

        # Cool down
        temperature *= COOLING_RATE

        # Progress update
        if (iteration + 1) % (MAX_ITERATIONS // 20) == 0:
            print(f"Iter: {iteration+1}/{MAX_ITERATIONS}, Temp: {temperature:.4f}, Cost: {current_cost:.4f}, Best Cost: {best_cost:.4f}")

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
    print(f"Final lowest cost found: {best_cost:.6f}")

    if best_cost < STOPPING_COST:
        print("A valid arrangement was likely found.")
        plot_arrangement(best_positions, radii, CONTAINER_WIDTH, CONTAINER_HEIGHT, f"Potential Solution (Cost: {best_cost:.6f})")
    else:
        print("Could not find a valid arrangement within the given iterations.")
        print("Displaying the best (lowest cost) arrangement found.")
        plot_arrangement(best_positions, radii, CONTAINER_WIDTH, CONTAINER_HEIGHT, f"Best Attempt (Cost: {best_cost:.6f})")

    return best_positions, best_cost


# --- Run the Solver ---
if __name__ == "__main__":
    final_positions, final_cost = solve_packing()
    # You can access the final coordinates in the 'final_positions' numpy array
    # print("\nFinal Positions (x, y) for each coin:")
    # for i in range(num_coins):
    #    print(f"  Coin {i} (Radius {radii[i]:.3f}): {final_positions[i]}")