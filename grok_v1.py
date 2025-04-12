import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Coin dimensions (radii in mm)
coins = {
    'penny': {'radius': 9.525, 'count': 4},   # 19.05 mm diameter
    'nickel': {'radius': 10.605, 'count': 1}, # 21.21 mm diameter
    'dime': {'radius': 8.955, 'count': 2},    # 17.91 mm diameter
    'quarter': {'radius': 12.13, 'count': 3}  # 24.26 mm diameter
}

# Credit card dimensions (mm)
CARD_WIDTH = 85.60
CARD_HEIGHT = 53.98

# Create list of coin radii based on counts
radii = []
for coin_type, info in coins.items():
    radii.extend([info['radius']] * info['count'])
radii = np.array(radii)  # [penny, penny, penny, penny, nickel, dime, dime, quarter, quarter, quarter]
N = len(radii)  # 10 coins

# Objective function: penalizes overlaps and boundary violations
def objective(positions, radii):
    penalty = 0
    positions = positions.reshape(-1, 2)  # N x 2 array of (x, y)
    
    # Check overlaps between all pairs
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2  # Penalize overlap
    
    # Check boundary violations
    for i in range(N):
        x, y = positions[i]
        r = radii[i]
        # Penalize if coin crosses boundaries
        if x - r < 0:
            penalty += (x - r)**2
        if x + r > CARD_WIDTH:
            penalty += (x + r - CARD_WIDTH)**2
        if y - r < 0:
            penalty += (y - r)**2
        if y + r > CARD_HEIGHT:
            penalty += (y + r - CARD_HEIGHT)**2
    
    return penalty

# Simulated annealing
def simulated_annealing(radii, max_iterations=10000, initial_temp=10.0, cooling_rate=0.995):
    # Initialize random positions within card bounds
    positions = np.zeros((N, 2))
    for i in range(N):
        r = radii[i]
        # Ensure coins start fully inside bounds
        positions[i, 0] = np.random.uniform(r, CARD_WIDTH - r)
        positions[i, 1] = np.random.uniform(r, CARD_HEIGHT - r)
    
    current_positions = positions.flatten()
    current_score = objective(current_positions, radii)
    best_positions = current_positions.copy()
    best_score = current_score
    
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # Generate new candidate by perturbing positions
        new_positions = current_positions.copy()
        i = np.random.randint(0, N * 2)  # Perturb one x or y coordinate
        r = radii[i // 2]
        # Limit perturbation to avoid jumping too far
        new_positions[i] += np.random.normal(0, min(CARD_WIDTH, CARD_HEIGHT) / 10)
        # Clamp to ensure coin stays roughly in bounds
        new_positions[i] = np.clip(new_positions[i], r, (CARD_WIDTH if i % 2 == 0 else CARD_HEIGHT) - r)
        
        new_score = objective(new_positions, radii)
        
        # Accept new solution based on Metropolis criterion
        if new_score <= current_score or np.random.random() < np.exp(-(new_score - current_score) / temp):
            current_positions = new_positions
            current_score = new_score
        
        # Update best solution
        if current_score < best_score:
            best_positions = current_positions.copy()
            best_score = current_score
        
        # Cool the temperature
        temp *= cooling_rate
        
        # Early stopping if solution is good enough (no overlaps or boundary issues)
        if best_score < 1e-6:
            break
    
    return best_positions.reshape(-1, 2), best_score

# Run the simulation
np.random.seed(42)  # For reproducibility
positions, final_score = simulated_annealing(radii)

# Visualize the result
fig, ax = plt.subplots()
ax.set_xlim(0, CARD_WIDTH)
ax.set_ylim(0, CARD_HEIGHT)
ax.set_aspect('equal')
ax.set_title(f'Coin Packing (Score: {final_score:.4f})')

# Draw credit card rectangle
rect = plt.Rectangle((0, 0), CARD_WIDTH, CARD_HEIGHT, fill=False, edgecolor='black')
ax.add_patch(rect)

# Draw coins
coin_colors = {'penny': 'red', 'nickel': 'green', 'dime': 'blue', 'quarter': 'purple'}
coin_index = 0
for coin_type, info in coins.items():
    for _ in range(info['count']):
        circle = Circle(positions[coin_index], radii[coin_index], color=coin_colors[coin_type], alpha=0.5, label=coin_type if coin_index == 0 or _ == 0 else "")
        ax.add_patch(circle)
        coin_index += 1

# Add legend
plt.legend()

# Print final positions and check feasibility
print("Final coin positions (x, y):")
for i in range(N):
    print(f"Coin {i+1} (radius {radii[i]:.3f}): ({positions[i, 0]:.2f}, {positions[i, 1]:.2f})")

print(f"\nFinal objective score: {final_score:.4f}")
if final_score < 1e-6:
    print("Success: Coins fit without overlap within the credit card area.")
else:
    print("Warning: Solution may have overlaps or boundary violations.")

plt.show()