import numpy as np
import math
import time
from objective_functions import ObjectiveFunctions

class DynamicProgrammingOptimizer:
    
    def __init__(self, objective_function: ObjectiveFunctions, total_wickets=10):
        self.objective_function = objective_function
        self.dp_table = {}
        self.optimal_decisions = {}
        self.num_states_evaluated = 0
        self.total_wickets = total_wickets

    def optimize(self, batsmen, bowling_probability_matrix, bowlers, num_overs=20):
        """
        Find the global optimal batting order using Dynamic Programming.
        """
        # Reset state
        self.dp_table = {}
        self.optimal_decisions = {}
        self.num_states_evaluated = 0
        
        # Update total_wickets based on the actual list provided
        # For 7 batsmen, max wickets = 6 (since 1 must remain not out)
        self.total_wickets = len(batsmen) - 1
        
        start_time = time.time()
        
        # Use frozenset for the state key
        all_batsmen_set = frozenset(batsmen)
        
        best_overall_value = -float('inf')
        best_opening_striker = None
        
        # Try every possible opening striker
        # For 7 batsmen, this loop runs 7 times.
        for opening_striker in batsmen:
            value = self.dp_solve(
                all_batsmen_set,
                0, # wickets_lost
                0, # over
                opening_striker,
                bowling_probability_matrix,
                bowlers,
                num_overs
            )
            
            if value > best_overall_value:
                best_overall_value = value
                best_opening_striker = opening_striker
                
        elapsed = time.time() - start_time
        print(f"DP Optimization finished in {elapsed:.2f} seconds.")
        print(f"States Evaluated: {self.num_states_evaluated}")
        
        # Reconstruct the optimal order from the policy
        optimal_order = self.reconstruct_order(
            all_batsmen_set,
            0,
            0,
            best_opening_striker,
            bowling_probability_matrix,
            bowlers,
            num_overs
        )
        
        return optimal_order, best_overall_value

    def dp_solve(self, remaining_batsmen, wickets_lost, over, striker, bowling_probability_matrix, bowlers, num_overs):
        """ 
        Recursive DP Solution. 
        Calculates max expected runs from state: (remaining, wickets, over, striker)
        """
        # --- BASE CASES ---
        if over >= num_overs:
            return 0.0
        
        if wickets_lost >= self.total_wickets:
            return 0.0
        
        if len(remaining_batsmen) < 2:
            return 0.0
        
        # --- MEMOIZATION ---
        state = (remaining_batsmen, wickets_lost, over, striker)
        if state in self.dp_table:
            return self.dp_table[state]
        
        self.num_states_evaluated += 1
        
        best_value = -float('inf')
        best_non_striker = None
        overall_best_next_batsman = None # To store the optimal 'next man in'
        
        # Iterate over all possible partners (decisions)
        for non_striker in remaining_batsmen:
            if non_striker == striker:
                continue
            
            # 1. Calculate current over outcomes
            over_runs, p_wicket, p_rotate = self.compute_over_value(
                striker, non_striker, bowling_probability_matrix[:, over], bowlers, wickets_lost
            )
            
            # 2. Calculate Future Value for 3 Scenarios
            
            # Case A: No Wicket, No Rotation
            prob_1 = (1 - p_wicket) * (1 - p_rotate)
            val_1 = self.dp_solve(remaining_batsmen, wickets_lost, over + 1, striker, 
                                  bowling_probability_matrix, bowlers, num_overs)
            
            # Case B: No Wicket, Rotation
            prob_2 = (1 - p_wicket) * p_rotate
            val_2 = self.dp_solve(remaining_batsmen, wickets_lost, over + 1, non_striker, 
                                  bowling_probability_matrix, bowlers, num_overs)
            
            # Case C: Wicket Falls
            prob_3 = p_wicket
            val_3 = 0.0
            best_next_batsman_for_this_ns = None
            
            if prob_3 > 0 and wickets_lost + 1 < self.total_wickets:
                remaining_after_wicket = remaining_batsmen - {striker}
                
                if len(remaining_after_wicket) >= 2:
                    # We must pick the best NEXT batsman to maximize future score
                    max_next_val = -float('inf')
                    for next_batsman in remaining_after_wicket:
                        if next_batsman == non_striker: continue
                        
                        v = self.dp_solve(remaining_after_wicket, wickets_lost + 1, over + 1, 
                                          non_striker, bowling_probability_matrix, bowlers, num_overs)
                        
                        if v > max_next_val:
                            max_next_val = v
                            best_next_batsman_for_this_ns = next_batsman
                    
                    val_3 = max_next_val if max_next_val > -float('inf') else 0.0

            # 3. Total Expected Value for this non_striker choice
            expected_future = (prob_1 * val_1) + (prob_2 * val_2) + (prob_3 * val_3)
            total_value = over_runs + expected_future
            
            # 4. Update Max
            if total_value > best_value:
                best_value = total_value
                best_non_striker = non_striker
                overall_best_next_batsman = best_next_batsman_for_this_ns

        # Store value and POLICY (Decision)
        self.dp_table[state] = best_value
        self.optimal_decisions[state] = (best_non_striker, overall_best_next_batsman)
        
        return best_value

    def compute_over_value(self, striker, non_striker, bowling_probs, bowlers, wickets_lost):
        """Helper to calculate expected runs, wicket prob, and rotation prob for an over."""
        total_runs = 0.0
        total_wicket_prob = 0.0
        total_rotation_prob = 0.0
        
        for i, bowler in enumerate(bowlers):
            p_bowler = bowling_probs[i]
            if p_bowler > 0:
                # Expected runs/wickets from objective function
                r, w = self.objective_function.compute_over_expected_runs(striker, non_striker, bowler, wickets_lost)
                total_runs += p_bowler * r
                total_wicket_prob += p_bowler * w
                
                # Rotation prob using Binomial
                probs = self.objective_function.get_run_probabilities(striker, bowler)
                p_rot_ball = probs.get(1, 0.0) + probs.get(3, 0.0)
                
                p_odd_rot = 0.0
                if 0.0 < p_rot_ball < 1.0:
                    p, q, n = p_rot_ball, 1.0 - p_rot_ball, 6
                    # Sum of P(k=1, 3, 5)
                    p_odd_rot = (math.comb(n,1) * p**1 * q**5) + \
                                (math.comb(n,3) * p**3 * q**3) + \
                                (math.comb(n,5) * p**5 * q**1)
                                
                total_rotation_prob += p_bowler * p_odd_rot
        
        total_wicket_prob = min(total_wicket_prob, 0.9)
        total_rotation_prob = min(max(total_rotation_prob, 0.0), 1.0)
        return total_runs, total_wicket_prob, total_rotation_prob

    def reconstruct_order(self, all_batsmen, wickets_lost, over, opening_striker, bowling_prob_matrix, bowlers, num_overs):
        """Reconstruct the optimal order by following the stored policy."""
        batting_order = [opening_striker]
        remaining = all_batsmen
        wickets = wickets_lost
        current_striker = opening_striker
        
        while over < num_overs and wickets < self.total_wickets and len(remaining) >= 2:
            state = (remaining, wickets, over, current_striker)
            
            if state not in self.optimal_decisions:
                break
            
            # RETRIEVE OPTIMAL DECISION
            best_non_striker, best_next_batsman = self.optimal_decisions[state]
            
            if best_non_striker not in batting_order:
                batting_order.append(best_non_striker)
            
            # Determine transition (Most likely outcome)
            _, p_wicket, p_rotate = self.compute_over_value(
                current_striker, best_non_striker, bowling_prob_matrix[:, over], bowlers, wickets
            )
            
            if p_wicket > 0.5:
                # Wicket Logic: Use the STORED best_next_batsman
                remaining = remaining - {current_striker}
                wickets += 1
                current_striker = best_non_striker
                
                if best_next_batsman and best_next_batsman not in batting_order:
                    batting_order.append(best_next_batsman)
                    
            elif p_rotate > 0.5:
                current_striker = best_non_striker
            
            over += 1
            
        # Append anyone left over
        for b in all_batsmen:
            if b not in batting_order:
                batting_order.append(b)
                
        return batting_order