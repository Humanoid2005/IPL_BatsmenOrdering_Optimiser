from objective_functions import ObjectiveFunctions

class GreedyOptimizer():
    
    def __init__(self, objective_functions: ObjectiveFunctions):
        self.obj_func = objective_functions
        
        if 'Role' in self.obj_func.batsman_profile.columns:
            self.roles = self.obj_func.batsman_profile['Role'].to_dict()
    
    def get_batsmen_by_role(self, available_batsmen):
        
        openers = []
        middle = []
        lower = []
        
        for batsman in available_batsmen:
            if batsman in self.obj_func.batsman_profile.index:
                role = self.obj_func.batsman_profile.loc[batsman, 'Role']
                
                if role == 'opener':
                    openers.append(batsman)
                elif role == 'middle':
                    middle.append(batsman)
                elif role == 'lower':
                    lower.append(batsman)
                else:
                    # Default to middle if unknown role
                    middle.append(batsman)
            else:
                # Default to lower if batsman not found
                lower.append(batsman)
        
        return {
            'opener': openers,
            'middle': middle,
            'lower': lower
        }
    
    def optimize(self, available_batsmen, bowling_probability_matrix, bowler_names, openers=None, middle_order=None, lower_order=None, num_overs=20):
        # Auto-detect roles if not provided
        if openers is None or middle_order is None or lower_order is None:
            
            roles_dict = self.get_batsmen_by_role(available_batsmen)
            openers = roles_dict['opener']
            middle_order = roles_dict['middle']
            lower_order = roles_dict['lower']
            

        batting_order = []
        
        # Phase 1: Select openers (positions 1-2)
        
        remaining_openers = openers.copy()
        for pos in range(min(2, len(remaining_openers))):
            if not remaining_openers:
                break
            best_opener = self.greedy_select_from_pool(
                batting_order,
                remaining_openers,
                available_batsmen,
                bowling_probability_matrix, # Passed Prob Matrix
                bowler_names,               # Passed Bowler Names
                num_overs,
                pos + 1
            )
            batting_order.append(best_opener)
            remaining_openers.remove(best_opener)
        
        # Phase 2: Select middle order (positions 3-7)
        
        remaining_middle = middle_order.copy()
        for pos in range(len(batting_order), min(7, len(batting_order) + len(remaining_middle))):
            if not remaining_middle:
                break
            best_middle = self.greedy_select_from_pool(
                batting_order,
                remaining_middle,
                available_batsmen,
                bowling_probability_matrix,
                bowler_names,
                num_overs,
                pos + 1
            )
            batting_order.append(best_middle)
            remaining_middle.remove(best_middle)
        
        # Phase 3: Select lower order (remaining positions)
        
        remaining_lower = lower_order.copy()
        while len(batting_order) < len(available_batsmen) and remaining_lower:
            best_lower = self.greedy_select_from_pool(
                batting_order,
                remaining_lower,
                available_batsmen,
                bowling_probability_matrix,
                bowler_names,
                num_overs,
                len(batting_order) + 1
            )
            batting_order.append(best_lower)
            remaining_lower.remove(best_lower)
        
        # Final evaluation using objective_hard (Probabilistic)
        final_score = self.obj_func.objective_function(
            batting_order,
            bowling_probability_matrix,
            bowler_names,
            num_overs
        )
        
        print(f"Final Order: {batting_order}")
        print(f"Expected Runs: {final_score:.2f}")
        
        return batting_order, final_score
    
    def heuristic_completion(self, remaining_batsmen):
        if not remaining_batsmen:
            return []
        
        # Get strike rates
        batsmen_with_sr = []
        for batsman in remaining_batsmen:
            if batsman in self.obj_func.batsman_profile.index:
                sr = self.obj_func.batsman_profile.loc[batsman, 'StrikeRate']
                batsmen_with_sr.append((batsman, sr))
            else:
                # Default strike rate if not found
                batsmen_with_sr.append((batsman, 100.0))
        
        # Sort by strike rate (descending)
        batsmen_with_sr.sort(key=lambda x: x[1], reverse=True)
        
        return [batsman for batsman, sr in batsmen_with_sr]
    
    def greedy_select_from_pool(self, current_order, candidate_pool, all_batsmen, bowling_probability_matrix, bowler_names, num_overs, position):
        best_batsman = None
        best_score = -float('inf')
        
        for candidate in candidate_pool:
            temp_order = current_order + [candidate]
            
            # Complete with remaining batsmen
            remaining = [b for b in all_batsmen if b not in temp_order]
            temp_order_complete = temp_order + self.heuristic_completion(remaining)
            
            # Use objective_hard for evaluation
            score = self.obj_func.objective_function(
                temp_order_complete,
                bowling_probability_matrix,
                bowler_names,
                num_overs
            )
            
            print(f"Position {position}: {candidate:20s} → {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_batsman = candidate
        
        print(f"Selected: {best_batsman} ({best_score:.2f})")
        
        return best_batsman