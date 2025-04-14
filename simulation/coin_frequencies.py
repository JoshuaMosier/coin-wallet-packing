import sys # Used only for Python version check below

def calculate_coin_usage_frequency(max_value=99):
    """
    Calculates the number of values (1 to max_value) for which each coin type
    is used at least once when making change with specific coin limits.

    Args:
        max_value (int): The maximum cent value to calculate change for (inclusive).

    Returns:
        dict: A dictionary with coin types ('Q', 'D', 'N', 'P') as keys
              and the count of values requiring that coin type as values.
    """

    # Define coin values in cents
    coin_values = {'Q': 25, 'D': 10, 'N': 5, 'P': 1}
    # Define available coins (the maximum allowed for ANY single transaction)
    # Constraint from the problem description
    available_coins_limit = {'Q': 3, 'D': 2, 'N': 1, 'P': 4}

    # Initialize counters for the number of VALUES requiring each coin type
    values_requiring_coin = {'Q': 0, 'D': 0, 'N': 0, 'P': 0}

    # List of coin types ordered from largest to smallest for the greedy algorithm
    coin_order = ['Q', 'D', 'N', 'P']

    # --- Simulation Loop ---
    # Loop through each value from 1 to max_value cents
    for value in range(1, max_value + 1):
        remaining_value = value
        # Keep track if a coin type was used for this specific value
        used_coin_type_this_value = {'Q': False, 'D': False, 'N': False, 'P': False}

        # --- Greedy Change Calculation for the current 'value' ---
        for coin_type in coin_order:
            value_of_coin = coin_values[coin_type]
            max_available_this_coin = available_coins_limit[coin_type]

            # Check if this coin can potentially be used
            if remaining_value >= value_of_coin and max_available_this_coin > 0:
                # Calculate how many of this coin *could* be used based on remaining value
                num_possible = remaining_value // value_of_coin
                # Calculate how many *can actually* be used based on availability limit
                num_to_use = min(num_possible, max_available_this_coin)

                # If at least one coin of this type was used, mark it and update remaining value
                if num_to_use > 0:
                    used_coin_type_this_value[coin_type] = True
                    remaining_value -= num_to_use * value_of_coin

        # --- Update Counters ---
        # After checking all coin types for this 'value', update the main counters
        # if a specific coin type was used at least once for this value.
        for coin_type in coin_order:
            if used_coin_type_this_value[coin_type]:
                values_requiring_coin[coin_type] += 1

        # Optional check: Ensure change was made correctly
        if remaining_value != 0:
            print(f"Error: Could not make exact change for {value} cents. Remaining: {remaining_value}")
            # This shouldn't happen with the given constraints for 1-99 cents

    return values_requiring_coin

# --- Main Execution ---
if __name__ == "__main__":
    # Basic check for Python 3
    if sys.version_info[0] < 3:
        print("This script requires Python 3.")
        sys.exit(1)

    # Run the calculation
    results = calculate_coin_usage_frequency(max_value=99)

    # Print the final results clearly
    print("--- Coin Usage Frequency (Based on Values Requiring Coin Type At Least Once) ---")
    print(f"Calculation Range: 1 to 99 cents")
    print(f"Coin Limits per Transaction: 3 Quarters, 2 Dimes, 1 Nickel, 4 Pennies")
    print("-" * 70)
    print(f"Number of values requiring at least one Quarter (Q): {results['Q']}")
    print(f"Number of values requiring at least one Dime    (D): {results['D']}")
    print(f"Number of values requiring at least one Nickel  (N): {results['N']}")
    print(f"Number of values requiring at least one Penny   (P): {results['P']}")
    print("-" * 70)