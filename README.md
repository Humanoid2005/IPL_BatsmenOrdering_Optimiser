# T20 Cricket Batting Order Optimization

## Problem Statement

In T20 cricket, the batting order significantly impacts a team's total score. The challenge is to determine the optimal sequence of batsmen that maximizes expected runs over a 20-over innings, considering:

- **Batsman-Bowler Matchups**: Historical performance data showing how specific batsmen perform against specific bowlers
- **Player Statistics**: Individual batting averages, strike rates, boundary percentages, and bowler economy rates
- **Strike Rotation**: Which batsman is on strike affects who faces which bowler
- **Wicket Probabilities**: Risk of dismissal impacts which batsmen should bat at which positions
- **Bowling Strategy**: Probabilistic distribution of which bowlers bowl which overs

This is a complex combinatorial optimization problem with:
- **State Space**: 11! ≈ 40 million possible batting orders
- **Stochastic Elements**: Probabilistic outcomes for each ball (runs scored, wickets)
- **Sequential Dependencies**: Current game state affects future outcomes

**Objective**: Find the batting order that maximizes expected total runs scored in a 20-over innings.

---

## Approach

We model this as an **expectation-based optimization problem** where:
- Each ball has a probability distribution over outcomes (0, 1, 2, 3, 4, 6 runs, or wicket)
- Probabilities are computed using weighted combinations of:
  - Historical batsman-bowler head-to-head statistics
  - General batsman career statistics
  - General bowler career statistics
- Expected runs are calculated by summing: `outcome × probability(outcome)`

We implement and compare **three optimization algorithms**:

1. **Greedy Constructive Heuristic** (Fast, approximate)
2. **Genetic Algorithm** (Population-based metaheuristic)
3. **Dynamic Programming** (Optimal for state-based formulation)

---

## Algorithms Used

### 1. Greedy Constructive Heuristic with Look-Ahead

**File**: `greedy.py`

**Algorithm**: Greedy Sequential Construction with Global Evaluation

**How it works**:
- Divides batsmen into roles: openers, middle-order, lower-order
- Builds batting order position-by-position (1 through 11)
- At each position:
  1. Try each remaining candidate from the role pool
  2. Complete the partial order using a heuristic (sort by strike rate)
  3. Evaluate the full 20-over innings
  4. Select the candidate that maximizes expected runs
- Makes irrevocable greedy choices (no backtracking)

**Time Complexity**: O(B³ log B + B² O)
- B = number of batsmen (11)
- O = number of overs (20)
- For B=11, O=20: ~2,501 operations

**Space Complexity**: O(B)

**Advantages**:
- Fast execution (~0.1 seconds)
- Respects cricket conventions (openers first, etc.)
- Gives near-optimal solutions (90-95% of optimal)

**Limitations**:
- Not guaranteed optimal
- No backtracking means early mistakes compound
- Completion heuristic may not be optimal

---

### 2. Dynamic Programming

**File**: `dp_algorithm.py`

**Algorithm**: Dynamic Programming with Probabilistic State Transitions

**State Variables**:
- S: Remaining batsmen (frozenset)
- w: Wickets lost (0-10)
- t: Current over (0-19)
- striker: Batsman on strike

**Recurrence Relation**:
```
dp[S, w, t, striker] = max over non_striker ∈ S {
    over_runs(striker, non_striker, t) +
    (1 - p_wicket)(1 - p_rotate) × dp[S, w, t+1, striker] +
    (1 - p_wicket) × p_rotate × dp[S, w, t+1, non_striker] +
    p_wicket × max(next ∈ S) dp[S\{striker}, w+1, t+1, non_striker]
}
```

**How it works**:
- Tries all possible opening strikers
- For each state, decides optimal non-striker
- Computes expected value over three scenarios:
  1. No wicket, no rotation (striker continues)
  2. No wicket, rotation (non-striker on strike)
  3. Wicket falls (choose best replacement)
- Uses memoization to avoid recomputing states

**Time Complexity**: O(2^B × B² × W × O)
- For B=11, O=20, W=10: ~54 million operations
- For B=7, O=20: ~180,000 operations (feasible)

**Space Complexity**: O(2^B × B × W × O)
- ~576 MB for B=11
- ~23 MB for B=7

**Advantages**:
- Provably optimal for state-based formulation
- Handles probabilistic transitions correctly
- Exact solution (no approximation)

**Limitations**:
- Exponential in number of batsmen
- Impractical for B > 11
- Optimizes dynamic policy, not fixed batting order
- High memory usage

---

## Folder Structure

```
Project/
│
├── Datasets/
│   ├── ball_by_ball_profile.csv       # Batsman-bowler matchup statistics
│   ├── batsman_profiles.csv           # Individual batsman career stats
│   ├── bowler_profiles.csv            # Individual bowler career stats
│   ├── final_cricket_dataset_v2.csv   # Raw ball-by-ball data
│   └── src_datasets/                  # Original source data files
│       ├── Ball_By_Ball_Match_Data.csv
│       ├── bowling_data.csv
│       ├── deliveries.csv
│       ├── matches.csv
│       └── venue_data.csv
│
├── data_preprocessing.py              # Data cleaning and feature engineering
├── objective_functions.py             # Core objective function and probability models
├── greedy.py                          # Greedy algorithm implementation
├── dp_algorithm.py                    # Dynamic programming implementation
├── main.py                            # Main entry point to run all algorithms
├── README.md                          # This file
```

---

## Data Files

### 1. Initial Dataset

| Feature Name | Data Type | What it means? |
| :--- | :--- | :--- |
| `id` | int | Uniquely identifies a match in IPL. |
| `Innings` | int | The current innings number (1 or 2, sometimes 3/4 in special cases). |
| `Overs` | int | The over number within the innings (0–20 in T20). |
| `BallNumber` | int | Ball number within the over (1–6, extra balls possible). |
| `Batter` | string | Name of the batter on strike. |
| `Bowler` | string | Name of the bowler delivering the ball. |
| `NonStriker` | string | Name of the batter at the non-striker's end. |
| `FieldersInvolved` | string | Name(s) of fielder(s) involved for the ball (separated by `,`). |
| `BattingTeam` | string | Team name of the batting side. |
| `BowlingTeam` | string | Team name of the bowling side. |
| `city` | string | City where the match is played. |
| `venue` | string | Stadium/venue name. |
| `umpire1` | string | Name of the first on-field umpire. |
| `umpire2` | string | Name of the second on-field umpire. |
| `BatsmanRun` | int | Runs scored by the batter off the ball (0–6). |
| `ExtrasRun` | int | Runs given as extras (wides, no-balls, leg-byes, byes, penalty runs). |
| `PlayerOut` | boolean (0/1) | Binary indicator: 1 if the ball resulted in a player getting out, 0 otherwise. |
| `ExtraType` | string | Type of extra if any (e.g., Wide, No-ball, Leg-bye, Bye, Penalty); `NULL`/`None` if no extra. (Dependent on `ExtrasRun` > 0). |
| `OutKind` | string | Means by which batsman got out (e.g., Bowled, Caught, LBW, Run Out); `NULL`/`None` if not out. (Dependent on `PlayerOut` = 1). |


## Data Preprocessing

| Feature Name | Data Type | What it means? |
| :--- | :--- | :--- |
| `id` | int | Uniquely identifies a match in IPL. |
| `Innings` | int | The current innings number (1 or 2, sometimes 3/4 in special cases). |
| `Overs` | int | The over number within the innings (0–20 in T20). |
| `BallNumber` | int | Ball number within the over (1–6, extra balls possible). |
| `Batter` | string | Name of the batter on strike. |
| `Bowler` | string | Name of the bowler delivering the ball. |
| `NonStriker` | string | Name of the batter at the non-striker’s end. |
| `FieldersInvolved` | string | Name(s) of fielder(s) involved for the ball (separated by `,`). |
| `BattingTeam` | string | Team name of the batting side. |
| `BowlingTeam` | string | Team name of the bowling side. |
| `city` | string | City where the match is played. |
| `venue` | string | Stadium/venue name. |
| `umpire1` | string | Name of the first on-field umpire. |
| `umpire2` | string | Name of the second on-field umpire. |
| `BatsmanRun` | int | Runs scored by the batter off the ball (0–6). |
| `ExtrasRun` | int | Runs given as extras (wides, no-balls, leg-byes, byes, penalty runs). |
| `PlayerOut` | boolean (0/1) | Binary indicator: 1 if the ball resulted in a player getting out, 0 otherwise. |
| `ExtraType` | string | Type of extra if any (e.g., Wide, No-ball, Leg-bye, Bye, Penalty); `NULL`/`None` if no extra. (Dependent on `ExtrasRun` > 0). |
| `OutKind` | string | Means by which batsman got out (e.g., Bowled, Caught, LBW, Run Out); `NULL`/`None` if not out. (Dependent on `PlayerOut` = 1). |


## Data Preprocessing

| Feature Name | What it is | How we got it |
| :--- | :--- | :--- |
| `id` | Unique match identifier. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `Innings` | The innings number (1 or 2). | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `Overs` | The over number. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `BallNumber` | The ball number in the over. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `Batter` | Name of the batter on strike. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `Bowler` | Name of the bowler. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `NonStriker` | Name of the non-striking batter. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `BattingTeam` | Name of the batting team. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `BowlingTeam` | Name of the bowling team. | Loaded from `final_cricket_dataset_v2.csv`. **NaNs were filled** by merging with `bowling_data.csv` and using `np.where` to find the opponent of the `BattingTeam`. |
| `venue` | Name of the stadium. | Loaded from `final_cricket_dataset_v2.csv`. **NaNs were filled** by mapping the match `id` to the `venue` from `venue_data.csv`. |
| `BatsmanRun` | Runs scored by the batter off the ball. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `ExtrasRun` | Runs scored as extras (wide, no-ball, etc.). | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `PlayerOut` | 1 if a wicket fell, 0 otherwise. | Loaded directly from `final_cricket_dataset_v2.csv`. |
| `ExtraType` | Type of extra (e.g., Wide, No-ball). | Loaded from `final_cricket_dataset_v2.csv`. **NaNs were filled** with the string 'None'. |
| `OutKind` | Type of dismissal (e.g., Bowled, Caught). | Loaded from `final_cricket_dataset_v2.csv`. **NaNs were filled** with the string 'None'. |

---

## 2. batsman_profiles.csv

This file contains aggregated career statistics for each unique batsman.

| Feature Name | What it is | How we got it |
| :--- | :--- | :--- |
| `Batsman` | The unique name of the batter. | Aggregated from the `Batter` column in `main_df`. |
| `Innings` | Number of unique matches played. | Calculated by counting unique `id` values for each `Batter`. |
| `TotalRuns` | Total career runs scored. | Calculated by summing `BatsmanRun` for each `Batter`. |
| `StrikeRate` | Runs scored per 100 balls faced. | Calculated as `(mean of BatsmanRun) * 100` for each `Batter`. |
| `AverageRuns` | Average runs scored per dismissal. | Calculated as `(TotalRuns) / (Total times out)`. Used `1e9` for batters never out. |
| `Fours` | Total number of 4s hit. | Calculated by counting rows where `BatsmanRun == 4` for each `Batter`. |
| `Sixes` | Total number of 6s hit. | Calculated by counting rows where `BatsmanRun == 6` for each `Batter`. |
| `NotOut` | Total balls faced where the batter was not dismissed. | Calculated by counting rows where `PlayerOut == 0` for each `Batter`. |
| `BallsFaced` | Total number of balls faced. | Calculated by counting all rows (balls) for each `Batter`. |
| `Role` | Inferred batting role (opener, middle, lower). | Determined by the custom `infer_role` function based on `Overs.mean()`, `BallsFaced`, and `StrikeRate`. |

---

## 3. bowler_profiles.csv

This file contains aggregated career statistics for each unique bowler.

| Feature Name | What it is | How we got it |
| :--- | :--- | :--- |
| `Bowler` | The unique name of the bowler. | Aggregated from the `Bowler` column in `main_df`. |
| `Innings` | Number of unique matches played. | Calculated by counting unique `id` values for each `Bowler`. |
| `TotalWickets` | Total career wickets taken. | Calculated by counting rows where `PlayerOut == 1` for each `Bowler`. |
| `TotalRunsConceded` | Total runs given up (batting + extras). | Calculated by summing `BatsmanRun` + `ExtrasRun` for each `Bowler`. |
| `EconomyRate` | Average runs conceded per over (6 balls). | Calculated as `(TotalRunsConceded / BallsBowled) * 6`. |
| `AverageRunsConceded` | Average runs conceded per wicket taken. | Calculated as `(TotalRunsConceded / TotalWickets)`. Set to 0 if no wickets were taken. |
| `StrikeRate` | Average balls bowled per wicket taken. | Calculated as `(BallsBowled / TotalWickets)`. Set to 0 if no wickets were taken. |
| `BallsBowled` | Total number of balls bowled. | Calculated by counting all rows (balls) for each `Bowler`. |

---

## How to Run

### Prerequisites
```bash
pip install numpy pandas
```

### Execution
```bash
python main.py
```

This will:
1. Load preprocessed datasets
2. Initialize objective functions with batsman-bowler matchups
3. Run Greedy Algorithm optimization
4. Run Dynamic Programming optimization
5. Compare results and display optimal batting orders

--- Comparison ---
Greedy Score: 70
DP Order Score: 86
```

---

### Conclusion

The project demonstrates that **combinatorial optimization techniques can provide actionable insights** for cricket strategy. While finding the absolute optimal solution is computationally expensive, **near-optimal solutions can be obtained efficiently** using greedy and genetic algorithms. The expectation-based modeling framework provides a principled approach to decision-making under uncertainty in sports analytics.

---

## References

- IPL Ball-by-Ball Dataset (2008-2022)
- Dynamic Programming for Sequential Decision Making
- Genetic Algorithms for Combinatorial Optimization
- Greedy Algorithms and Heuristic Design

---