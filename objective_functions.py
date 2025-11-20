import pandas as pd
import numpy as np

class ObjectiveFunctions:
    
    def __init__(self,ball_by_ball_path,batsman_profile_path,bowler_profile_path):
        self.ball_by_ball = pd.read_csv(ball_by_ball_path)
        self.batsman_profile = pd.read_csv(batsman_profile_path).set_index('Batsman')
        self.bowler_profile = pd.read_csv(bowler_profile_path).set_index('Bowler')
        
        self.matchup_map = self.build_matchup_map()
        
    def build_matchup_map(self):
        matchup_stats = {}
        
        grouped = self.ball_by_ball.groupby(['Batter', 'Bowler'])
        
        for (batsman, bowler), group in grouped:
            total_balls = len(group)
            total_runs = group['BatsmanRun'].sum()
            wickets = group['PlayerOut'].notna().sum()
            
            run_counts = group['BatsmanRun'].value_counts().to_dict()
            
            matchup_stats[(batsman, bowler)] = {
                'balls': total_balls,
                'runs': total_runs,
                'wickets': wickets,
                'avg_runs_per_ball': total_runs / total_balls if total_balls > 0 else 0,
                'wicket_prob_per_ball': wickets / total_balls if total_balls > 0 else 0,
                'run_distribution': run_counts
            }
        
        return matchup_stats
    
    def get_run_probabilities(self,batsman,bowler):
        matchup_key = (batsman, bowler)
        has_matchup = matchup_key in self.matchup_map
        
        if has_matchup:
            matchup = self.matchup_map[matchup_key]
            matchup_weight = min(matchup['balls'] / 50.0, 0.6)
        else:
            matchup_weight = 0.0
        
        general_weight = 1.0 - matchup_weight
        bat_weight = general_weight * 0.5
        bowl_weight = general_weight * 0.5
        
        if batsman in self.batsman_profile.index:
            bat_stats = self.batsman_profile.loc[batsman]
            bat_sr = bat_stats['StrikeRate'] / 100.0
            bat_balls = bat_stats['BallsFaced']
            bat_fours = bat_stats['Fours']
            bat_sixes = bat_stats['Sixes']
            
            p_four_bat = bat_fours / bat_balls if bat_balls > 0 else 0.10
            p_six_bat = bat_sixes / bat_balls if bat_balls > 0 else 0.04
        else:
            bat_sr = 1.0
            p_four_bat = 0.10
            p_six_bat = 0.04
        
        if bowler in self.bowler_profile.index:
            bowl_stats = self.bowler_profile.loc[bowler]
            bowl_er = bowl_stats['EconomyRate'] / 6.0
            bowl_sr = bowl_stats['StrikeRate']
            bowl_wicket_prob = 1.0 / bowl_sr if bowl_sr > 0 else 0.033
        else:
            bowl_er = 1.0
            bowl_wicket_prob = 0.033
        
        if has_matchup:
            expected_runs = (
                matchup_weight * matchup['avg_runs_per_ball'] +
                bat_weight * bat_sr +
                bowl_weight * bowl_er
            )
            wicket_prob = (
                matchup_weight * matchup['wicket_prob_per_ball'] +
                general_weight * bowl_wicket_prob
            )
        else:
            expected_runs = bat_weight * bat_sr + bowl_weight * bowl_er
            wicket_prob = bowl_wicket_prob
        
        wicket_prob = min(wicket_prob, 0.15)
        
        p_dot = 0.30
        p_four = p_four_bat
        p_six = p_six_bat
        
        remaining = 1.0 - p_dot - p_four - p_six - wicket_prob
        remaining = max(0, remaining)
        
        p_one = 0.50 * remaining
        p_two = 0.35 * remaining
        p_three = 0.15 * remaining
        
        total_prob = p_dot + p_one + p_two + p_three + p_four + p_six + wicket_prob
        
        if total_prob > 0:
            normalization = 1.0 / total_prob
            p_dot = p_dot/total_prob
            p_one = p_one/total_prob
            p_two = p_two/total_prob
            p_three = p_three/total_prob
            p_four = p_four/total_prob
            p_six = p_six/total_prob
            wicket_prob = wicket_prob/total_prob
        
        return {
            0: p_dot,
            1: p_one,
            2: p_two,
            3: p_three,
            4: p_four,
            6: p_six,
            'wicket': wicket_prob
        }
    
    def compute_over_expected_runs(self,striker,non_striker,bowler,wickets_lost):
        over_runs = 0.0
        over_wickets = 0.0
        
        current_striker = striker
        current_non_striker = non_striker
        
        for ball in range(6):
            probs = self.get_run_probabilities(current_striker, bowler)
            
            ball_runs = (
                0 * probs[0] +
                1 * probs[1] +
                2 * probs[2] +
                3 * probs[3] +
                4 * probs[4] +
                6 * probs[6]
            )
            
            over_runs += ball_runs
            over_wickets += probs['wicket']
            
            p_rotate = probs[1] + probs[3]
            p_wicket = probs['wicket']
            
            if p_wicket > 0.5:
                pass
            elif p_rotate > 0.5:
                current_striker, current_non_striker = current_non_striker, current_striker
        
        return over_runs, over_wickets
    
    def objective_function(self,batting_order,bowling_probability_matrix,bowler_names,num_overs=20):
        if len(batting_order) < 2:
            return 0.0
        
        # Convert to numpy array if it's a list
        if isinstance(bowling_probability_matrix, list):
            bowling_probability_matrix = np.array(bowling_probability_matrix)
            
        max_wickets = len(batting_order) - 1
        num_bowlers = len(bowler_names)
        
        if bowling_probability_matrix.shape != (num_bowlers, num_overs):
            raise ValueError(
                f"Probability matrix shape {bowling_probability_matrix.shape} "
                f"doesn't match ({num_bowlers}, {num_overs})"
            )
        
        for k in range(num_overs):
            prob_sum = bowling_probability_matrix[:, k].sum()
            if not np.isclose(prob_sum, 1.0, atol=0.01):
                print(f"Warning: Probabilities for over {k+1} sum to {prob_sum}, not 1.0")

        for bowler_idx, bowler in enumerate(bowler_names):
            total_prob_overs = bowling_probability_matrix[bowler_idx, :].sum()
            if total_prob_overs > 4.0 + 1e-6:  # Allow small floating-point tolerance
                raise ValueError(f"Bowler {bowler} has an expected {total_prob_overs} overs, which exceeds the limit of 4.")

        total_runs = 0.0
        expected_wickets = 0.0
        
        for over_num in range(num_overs):
            wickets_lost = int(min(expected_wickets, max_wickets - 1))
            
            striker_idx = wickets_lost
            non_striker_idx = wickets_lost + 1
            
            if striker_idx >= len(batting_order) or non_striker_idx >= len(batting_order):
                break
            
            striker = batting_order[striker_idx]
            non_striker = batting_order[non_striker_idx]
            
            over_expected_runs = 0.0
            over_expected_wickets = 0.0
            
            for bowler_idx, bowler in enumerate(bowler_names):
                prob = bowling_probability_matrix[bowler_idx, over_num]
                
                if prob > 0:
                    runs_vs_bowler, wickets_vs_bowler = self.compute_over_expected_runs(
                        striker, non_striker, bowler, wickets_lost
                    )
                    
                    over_expected_runs += prob * runs_vs_bowler
                    over_expected_wickets += prob * wickets_vs_bowler
            
            total_runs += over_expected_runs
            expected_wickets += over_expected_wickets
            
            if expected_wickets >= max_wickets:
                break
        
        return total_runs


def create_bowling_probability_matrix(bowler_names,num_overs= 20,strategy='balanced'):
    
    num_bowlers = len(bowler_names)
    P = np.zeros((num_bowlers, num_overs))
    
    if strategy == 'balanced':
        for k in range(num_overs):
            if k < 6:
                probs = np.random.dirichlet(np.ones(num_bowlers) * 0.5)
            elif k < 16:
                probs = np.random.dirichlet(np.ones(num_bowlers) * 0.8)
            else:
                probs = np.random.dirichlet(np.ones(num_bowlers) * 0.5)
            
            P[:, k] = probs
    
    elif strategy == 'uniform':
        P = np.ones((num_bowlers, num_overs)) / num_bowlers
    
    else:
        for k in range(num_overs):
            probs = np.random.dirichlet(np.ones(num_bowlers))
            P[:, k] = probs
    
    # Iteratively enforce 4-over constraint and column normalization
    max_iterations = 100
    for iteration in range(max_iterations):
        # Step 1: Enforce 4-over constraint for each bowler
        constraint_violated = False
        for bowler_idx in range(num_bowlers):
            total_overs = P[bowler_idx, :].sum()
            if total_overs > 4.0 + 1e-6:  # Small tolerance
                P[bowler_idx, :] = (P[bowler_idx, :] / total_overs) * 4.0
                constraint_violated = True
        
        # Step 2: Renormalize each over to sum to 1
        for k in range(num_overs):
            over_sum = P[:, k].sum()
            if over_sum > 1e-9:
                P[:, k] = P[:, k] / over_sum
            else:
                # If column is all zeros, assign to random bowler
                P[np.random.randint(0, num_bowlers), k] = 1.0
        
        # Check convergence
        if not constraint_violated:
            # Verify all constraints are satisfied
            all_satisfied = True
            for bowler_idx in range(num_bowlers):
                if P[bowler_idx, :].sum() > 4.0 + 1e-6:
                    all_satisfied = False
                    break
            if all_satisfied:
                break
    
    # Final verification and capping (safety measure)
    for bowler_idx in range(num_bowlers):
        total_overs = P[bowler_idx, :].sum()
        if total_overs > 4.0:
            # Hard cap: scale down to exactly 4.0
            P[bowler_idx, :] = (P[bowler_idx, :] / total_overs) * 3.99999
    
    # Final column normalization
    for k in range(num_overs):
        over_sum = P[:, k].sum()
        if over_sum > 1e-9:
            P[:, k] = P[:, k] / over_sum
            
    return P
