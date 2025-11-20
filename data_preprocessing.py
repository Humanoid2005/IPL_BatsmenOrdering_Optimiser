import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

main_df = pd.read_csv('Datasets/final_cricket_dataset_v2.csv')
venue_df = pd.read_csv('Datasets/src_datasets/venue_data.csv')
bowling_df = pd.read_csv('Datasets/src_datasets/bowling_data.csv')

cols_to_drop = ['FieldersInvolved', 'umpire1', 'umpire2', 'city']
main_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
main_df['ExtraType'].fillna('None', inplace=True)
main_df['OutKind'].fillna('None', inplace=True)

# Fill 'venue' using a fast map-based lookup
# Create a dictionary-like object (a Series) mapping each match 'id' to its 'venue'
venue_map = venue_df.set_index('id')['venue']

# Use .map() to look up the venue for each 'id' in main_df and fill only the NaN values
main_df['venue'].fillna(main_df['id'].map(venue_map), inplace=True)


# Fill 'BowlingTeam' using a merge and np.where
# First, merge main_df with bowling_df to get team1 and team2 for each match
# This adds the opponent information to every row of main_df
merged_df = pd.merge(
    main_df,
    bowling_df[['id', 'team1', 'team2']], # Only need these columns from bowling_df
    on='id',
    how='left'
)

# Now, use the vectorized np.where to determine the BowlingTeam
# Condition: If team1 is the BattingTeam, the BowlingTeam is team2. Otherwise, it's team1.
calculated_bowling_teams = np.where(
    merged_df['team1'] == merged_df['BattingTeam'], # Condition
    merged_df['team2'],                              # Value if True
    merged_df['team1']                               # Value if False
)

# Fill the NaN values in the original 'BowlingTeam' column with our calculated values
main_df['BowlingTeam'].fillna(pd.Series(calculated_bowling_teams, index=main_df.index), inplace=True)

main_df.to_csv("Datasets/ball_by_ball_profile.csv", index=False)

# Generate Batsman Profiles

def infer_role(batsman):
    batsman_data = main_df[main_df['Batter'] == batsman]
    
    if len(batsman_data) == 0:
        return 'lower'
    
    avg_over = batsman_data['Overs'].mean()
    total_balls = len(batsman_data)
    strike_rate = batsman_profile_df[batsman_profile_df['Batsman'] == batsman]['StrikeRate'].values[0]
    
    if avg_over <= 5 and total_balls >= 100:
        return 'opener'
    elif avg_over <= 10 and strike_rate >= 120:
        return 'middle'
    elif strike_rate < 100 or total_balls < 50:
        return 'lower'
    else:
        return 'middle'

batsmen = main_df['Batter'].unique()
batsmen_matches_played = [len(main_df[main_df['Batter'] == batsman]['id'].unique()) for batsman in batsmen]
batsmen_innings = [len(main_df[main_df['Batter'] == batsman]['id'].unique()) for batsman in batsmen]
batsmen_total_runs = [main_df[main_df['Batter'] == batsman]['BatsmanRun'].sum() for batsman in batsmen]
batsmen_strike_rate = [(main_df[main_df['Batter'] == batsman]['BatsmanRun'].mean())*100 for batsman in batsmen]
batsmen_average_runs = [(main_df[main_df['Batter'] == batsman]['BatsmanRun'].sum())/(len(main_df[(main_df['Batter'] == batsman) & (main_df['PlayerOut']==1)])) if (len(main_df[(main_df['Batter'] == batsman) & (main_df['PlayerOut']==1)]))>0 else 1e9 for batsman in batsmen]
batsmen_fours = [len(main_df[(main_df['Batter'] == batsman) & (main_df['BatsmanRun'] == 4)]) for batsman in batsmen]
batsmen_sixes = [len(main_df[(main_df['Batter'] == batsman) & (main_df['BatsmanRun'] == 6)]) for batsman in batsmen]
batsmen_notout = [len(main_df[(main_df['Batter'] == batsman) & (main_df['PlayerOut'] == 0)]) for batsman in batsmen]
batsmen_balls_faced = [len(main_df[main_df['Batter'] == batsman]) for batsman in batsmen]

batsman_profile_df = pd.DataFrame({
    'Batsman': batsmen,
    'Innings': list(batsmen_innings),
    'TotalRuns': list(batsmen_total_runs),
    'StrikeRate': list(batsmen_strike_rate),
    'AverageRuns': list(batsmen_average_runs),
    'Fours': list(batsmen_fours),
    'Sixes': list(batsmen_sixes),
    'NotOut': list(batsmen_notout),
    'BallsFaced': list(batsmen_balls_faced)
})
batsman_profile_df['Role'] = batsman_profile_df['Batsman'].apply(infer_role)

batsman_profile_df.to_csv("Datasets/batsman_profiles.csv", index=False)

# Generate Bowler Profiles

bowlers = main_df['Bowler'].unique()
bowlers_innings = [len(main_df[main_df['Bowler'] == bowler]['id'].unique()) for bowler in bowlers]
bowlers_total_wickets = [len(main_df[(main_df['Bowler'] == bowler) & (main_df['PlayerOut'] == 1)]) for bowler in bowlers]
bowlers_total_runs_conceded = [main_df[main_df['Bowler'] == bowler]['BatsmanRun'].sum() + main_df[main_df['Bowler'] == bowler]['ExtrasRun'].sum() for bowler in bowlers]
bowlers_economy_rate = [((main_df[main_df['Bowler'] == bowler]['BatsmanRun'].sum() + main_df[main_df['Bowler'] == bowler]['ExtrasRun'].sum()) / len(main_df[main_df['Bowler'] == bowler])) * 6 for bowler in bowlers]
bowlers_average_runs_conceded = [((main_df[main_df['Bowler'] == bowler]['BatsmanRun'].sum() + main_df[main_df['Bowler'] == bowler]['ExtrasRun'].sum()) / len(main_df[(main_df['Bowler'] == bowler) & (main_df['PlayerOut'] == 1)])) if len(main_df[(main_df['Bowler'] == bowler) & (main_df['PlayerOut'] == 1)]) > 0 else 0 for bowler in bowlers]
bowlers_strike_rate = [(len(main_df[main_df['Bowler'] == bowler]) / len(main_df[(main_df['Bowler'] == bowler) & (main_df['PlayerOut'] == 1)])) if len(main_df[(main_df['Bowler'] == bowler) & (main_df['PlayerOut'] == 1)]) > 0 else 0 for bowler in bowlers]
bowlers_balls_bowled = [len(main_df[main_df['Bowler'] == bowler]) for bowler in bowlers]

bowler_profile_df = pd.DataFrame({
    'Bowler': bowlers,
    'Innings': list(bowlers_innings),
    'TotalWickets': list(bowlers_total_wickets),
    'TotalRunsConceded': list(bowlers_total_runs_conceded),
    'EconomyRate': list(bowlers_economy_rate),
    'AverageRunsConceded': list(bowlers_average_runs_conceded),
    'StrikeRate': list(bowlers_strike_rate),
    'BallsBowled': list(bowlers_balls_bowled)
})

bowler_profile_df.to_csv("Datasets/bowler_profiles.csv", index=False)