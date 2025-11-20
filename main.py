import numpy as np
import pandas as pd
import os
import random
from objective_functions import ObjectiveFunctions, create_bowling_probability_matrix
from dp_algorithm import DynamicProgrammingOptimizer
from greedy import GreedyOptimizer

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    ball_by_ball_path = os.path.join(current_dir, "Datasets", "ball_by_ball_profile.csv")
    batsman_profile_path = os.path.join(current_dir, "Datasets", "batsman_profiles.csv")
    bowler_profile_path = os.path.join(current_dir, "Datasets", "bowler_profiles.csv")
    
    print(ball_by_ball_path)
    
    print("Initializing Objective Functions...")
    try:
        objective_function = ObjectiveFunctions(
            ball_by_ball_path,
            batsman_profile_path,
            bowler_profile_path
        )
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure the CSV files exist in the 'Datasets' folder.")
        return

    available_batsmen = [
        "R Sharma",
        "V Kohli",
        "AB de Villiers",
        "MS Dhoni",
        "KA Pollard",
        "Shubman Gill",
        "KL Rahul"
    ]
    
    bowler_names = [
        "JJ Bumrah", 
        "TA Boult",
        "Rashid Khan",
        "YS Chahal",
        "HH Pandya",
        "RA Jadeja"
    ]
    
    num_overs = 20
    
    P = create_bowling_probability_matrix(bowler_names, num_overs, strategy='balanced')
    
    print("RUNNING OPTIMIZER 1: GREEDY")
    greedy_optimiser = GreedyOptimizer(objective_functions=objective_function)
    batting_order, final_score = greedy_optimiser.optimize(
        available_batsmen=available_batsmen,
        bowling_probability_matrix=P,
        bowler_names=bowler_names,
        num_overs=num_overs
    )
    
    print("\n--- Greedy Results ---")
    print(f"Score: {final_score:.2f}")
    print("Batting Order:")
    for i, batsman in enumerate(batting_order):
        print(f"  {i+1}. {batsman}")

    print("RUNNING OPTIMIZER 2: DYNAMIC PROGRAMMING")
    
    total_wickets = len(available_batsmen) - 1
        
    dp_optimizer = DynamicProgrammingOptimizer(
        objective_function=objective_function, 
        total_wickets=total_wickets
    )
    
    dp_best_order, dp_best_score = dp_optimizer.optimize(
        batsmen=available_batsmen,
        bowling_probability_matrix=P,
        bowlers=bowler_names,
        num_overs=num_overs
    )
    
    # Evaluate the DP order using the same objective function as GA
    dp_actual_score = objective_function.objective_function(
        dp_best_order,
        P,
        bowler_names,
        num_overs
    )
    
    print("\n--- Dynamic Programming Results ---")
    print(f"DP Score: {dp_actual_score:.2f}")
    print("Optimal Policy Batting Order:")
    for i, batsman in enumerate(dp_best_order):
        print(f"  {i+1}. {batsman}")

if __name__ == "__main__":
    main()