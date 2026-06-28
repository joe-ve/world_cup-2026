import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIGURATION & DATA LOADING ---
def get_data():
    # Load both sheets from your single Excel file
    file_path = 'C:/vs/world_cup/matches.xlsx'
    hist = pd.read_excel(file_path, sheet_name='Historic_Data')
    wc26 = pd.read_excel(file_path, sheet_name='WC_2026')
    return hist, wc26

# --- 2. FEATURE ENGINEERING ---
def engineer_features(df):
    # Calculate the Elo difference
    df['Elo_Diff'] = df['Home_Elo'] - df['Away_Elo']
    
    # Fill in any missing stages (e.g., historical friendlies)
    df['Tournament_Stage'] = df['Tournament_Stage'].fillna(0)
    
    # Future-proofing: Add logic here if you want to calculate rolling averages
    # for now, we use the core features
    return df

# --- 3. MAIN PREDICTION PIPELINE ---
def main():
    # Load and Prepare Data
    hist, wc26 = get_data()

    # Combine History + Finished Tournament matches for Training
    wc26_finished = wc26.dropna(subset=['Outcome'])
    train_data = pd.concat([hist, wc26_finished], ignore_index=True)

    # Create a copy of the training data
    df_swapped = train_data.copy()

# Swap Home and Away columns
    df_swapped.rename(columns={
        'Home_Team': 'Away_Team', 'Away_Team': 'Home_Team',
        'Home_Elo': 'Away_Elo', 'Away_Elo': 'Home_Elo',
        'Home_Goals': 'Away_Goals', 'Away_Goals': 'Home_Goals'
    }, inplace=True)

# Swap the Outcome: If it was Home Win (2), it becomes Away Win (0). If Draw (1), stays (1).
    df_swapped['Outcome'] = df_swapped['Outcome'].replace({2: 0, 0: 2})

# Combine original with the swapped data
    train_data = pd.concat([train_data, df_swapped], ignore_index=True)
    
    # Apply Feature Engineering
    train_data = engineer_features(train_data)
    
    # Train the Model
    features = ['Elo_Diff', 'Tournament_Stage', 'Is_Neutral']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    model.fit(train_data[features], train_data['Outcome'])
    
    print("Model trained successfully on", len(train_data), "matches.")
    
    # --- 4. INTERACTIVE QUESTIONNAIRE ---
    while True:
        print("\n--- Match Prediction ---")
        home = input("Home Team: ")
        away = input("Away Team: ")
        h_elo = float(input(f"Enter {home} Elo: "))
        a_elo = float(input(f"Enter {away} Elo: "))
        stage = int(input("Tournament Stage (1=Group, 2=Round of 32, 3=Round of 16, 4=Quarter-Finals, 5=Semi-Finals, 6=Final): "))
        
        # Format input for the model
        input_data = pd.DataFrame([[h_elo - a_elo, stage, 1]], columns=features)
        
        # Predict
        probs = model.predict_proba(input_data)[0]

        home_win_chance = probs[2] + (probs[1] / 2)
        away_win_chance = probs[0] + (probs[1] / 2)
        
        print(f"\nResults for {home} vs {away}:")
        print(f"Home Win: {home_win_chance:.2%}")
        print(f"Away Win: {away_win_chance:.2%}")
        
        if input("\nPredict another? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()