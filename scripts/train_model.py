
"""
ATP Tennis Player Development - Multi-Class Tier Prediction
Complete Training Pipeline

This script follows the notebook workflow exactly:
1. Load data and create multi-class target
2. Comprehensive feature engineering (demographics, psychology, performance, momentum)
3. RFE-CV feature selection
4. Hyperparameter tuning
5. Model evaluation with outputs

Usage:
    python train_model.py

Outputs:
    - Console: Top 15 features + performance metrics
    - outputs/confusion_matrices.png
    - outputs/feature_importance_comparison.png
    - outputs/classification_reports.txt
"""

#### ============================================================================
# ATP Tennis Player Development - Multi-Class Tier Prediction
# ============================================================================
"""
Project: Predicting tennis player development trajectories

Problem Evolution:
- Initial: Binary classification (top 50 vs rest) → 3% positive cases, weak signal
- Discovery: Investigation revealed gradient - 17% reached top 150, 59% improved 100+ spots  
- Solution: Multi-class classification capturing 4 development tiers

Tiers:
4 - Breakthrough (0-99): ~10 players
3 - Rising (100-199): ~54 players
2 - Challenger (200-299): ~58 players
1 - No Progress (300+): ~171 players

Features: Performance vs higher-ranked opponents + win consistency
Timeline: 2021 features → 2022-2023 outcomes
"""

# Imports
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import logging
import time
from pathlib import Path

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*ChainedAssignment.*', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Set random seed for reproducibility
np.random.seed(42)

# Create outputs directory
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("✅ Libraries loaded")
print("📊 Ready to build multi-class tier prediction model")

# ============================================================================
# SECTION 1: Data Loading & Setup
# ============================================================================

print(f"\n{'='*80}")
print("🔧 SECTION 1: Data Loading")
print("="*80)

# Load feature data
df = pd.read_csv('tennis_breakthrough_dataset.csv')
print(f"Loaded features for {len(df)} players")

# Connect to database for ranking outcomes
conn = sqlite3.connect('tennis_breakthrough.db')

# Get player outcomes (from previous investigation)
player_ids_str = ','.join(map(str, df['player_id'].tolist()))

# Step 1: Get baseline rankings (September 2021) for our players
print("\n📊 Step 1: Getting September 2021 baseline rankings...")
print("-"*80)

initial_query = f"""
SELECT
    player_id,
    MIN(rank) as baseline_rank,
    ranking_date as baseline_date
FROM rankings
WHERE ranking_date BETWEEN 20210901 AND 20210930
  AND player_id IN ({player_ids_str})
GROUP BY player_id
"""

baseline_df = pd.read_sql_query(initial_query, conn)

# Step 2: Get best ranking achieved (Jan 2022 - July 2023)
print("\n📊 Step 2: Getting best rankings achieved Jan 2022 - Jan 2023...")
print("-"*80)

best_rank_query = f"""
SELECT 
    player_id,
    MIN(rank) as best_rank,
    ranking_date as best_rank_date
FROM rankings
WHERE ranking_date BETWEEN 20220101 AND 20230131
  AND player_id IN ({player_ids_str})
GROUP BY player_id
"""

best_rank_df = pd.read_sql_query(best_rank_query, conn)
print(f"Found best rankings for {len(best_rank_df)} players")

# Merge outcomes
df_with_outcomes = df.copy().merge(baseline_df, on='player_id', how='inner')
df_with_outcomes = df_with_outcomes.merge(best_rank_df, on='player_id', how='left').copy()


print(f"\n✅ Combined dataset: {len(df_with_outcomes)} players")
print(f"   With baseline rank: {df_with_outcomes['baseline_rank'].notna().sum()}")
print(f"   With best rank: {df_with_outcomes['best_rank'].notna().sum()}")

# ============================================================================
# SECTION 2: Create Multi-Class Target & Correlation Analysis
# ============================================================================

print(f'\n{"="*80}')
print("🎯 SECTION 2: Creating Multi-Class Target")
print("="*80)

def classify_tier(best_rank):
    """Classify player into one of 4 tiers based on best ranking achieved."""
    if pd.isna(best_rank):
        return 1  # No_Progress
    elif best_rank <= 100:
        return 4  # Breakthrough
    elif best_rank <= 200:
        return 3  # Rising
    elif best_rank <= 300:
        return 2  # Challenger
    else:
        return 1  # No_Progress

df_with_outcomes['tier'] = df_with_outcomes['best_rank'].apply(classify_tier)
df_with_outcomes['tier_numeric'] = df_with_outcomes['tier']

# Distribution
tier_dist = df_with_outcomes['tier'].value_counts().sort_index()
print("\n📊 Target Distribution:")
print("-"*80)
tier_labels = {
    1: 'No_Progress (301+)',
    2: 'Challenger (201-300)', 
    3: 'Rising (101-200)',
    4: 'Breakthrough (1-100)'
}
for tier, count in tier_dist.items():
    pct = count / len(df_with_outcomes) * 100
    print(f"   {tier_labels[tier]}: {count} players ({pct:.1f}%)")

# ============================================================================
# SECTION 3: Comprehensive Feature Engineering
# ============================================================================

print(f"\n{'='*80}")
print("🔧 SECTION 3: Feature Engineering")
print("="*80)

# ============================================================================
# Part 1: Add Demographic Features
# ============================================================================

print("\n📊 Part 1: Adding Demographic Features...")
print("-"*80)

player_ids = df_with_outcomes['player_id'].tolist()
player_ids_str = ','.join(str(x) for x in player_ids)

demographics_query = f"""
SELECT 
    pid as player_id,
    Birth_Year,
    Nationality,
    Dominant_Hand,
    Height
FROM Players
WHERE pid IN ({player_ids_str})
"""

demographics_df = pd.read_sql_query(demographics_query, conn)
print(f"Found demographics for {len(demographics_df)} players")

# Merge with suffixes to handle any duplicates
df_with_outcomes = df_with_outcomes.merge(
    demographics_df,
    on='player_id',
    how='left',
    suffixes=('', '_demo')
).copy()

# Create derived demographic features
df_with_outcomes['age_in_2021'] = 2021 - df_with_outcomes['Birth_Year']
df_with_outcomes['is_lefty'] = (df_with_outcomes['Dominant_Hand'] == 'L').astype(int)
df_with_outcomes['height_cm'] = pd.to_numeric(df_with_outcomes['Height'], errors='coerce')

top_nations = ['ESP', 'FRA', 'ITA', 'ARG', 'USA', 'GER', 'RUS', 'AUS']
df_with_outcomes['from_top_nation'] = df_with_outcomes['Nationality'].isin(top_nations).astype(int)

print("✅ Added: age_in_2021, is_lefty, height_cm, from_top_nation")

# ============================================================================
# Part 2: Match Psychology Features
# ============================================================================

print("\n📊 Part 2: Creating Match Psychology Features...")
print("-"*80)

# Get match durations
match_duration_query = f"""
SELECT 
    pid as player_id,
    Won as result,
    duration as minutes
FROM Matches
WHERE Date BETWEEN 20210901 AND 20211231
  AND pid IN ({player_ids_str})
  AND Duration IS NOT NULL
  AND Duration > 0
  AND Duration != ''
"""

match_durations = pd.read_sql_query(match_duration_query, conn)
print(f"Found {len(match_durations)} matches with duration data")

# Build psychology features
psych_data = {
    'player_id': [],
    'avg_win_duration': [],
    'avg_loss_duration': [],
    'fight_score': [],
    'dominance_score': []
}

for player_id in df_with_outcomes['player_id'].unique():
    player_data = match_durations[match_durations['player_id'] == player_id]

    if len(player_data) == 0:
        psych_data['player_id'].append(int(player_id))
        psych_data['avg_win_duration'].append(None)
        psych_data['avg_loss_duration'].append(None)
        psych_data['fight_score'].append(None)
        psych_data['dominance_score'].append(None)
        continue

    # Convert to lists, filtering out invalid values
    results = []
    durations = []

    for _, row in player_data.iterrows():
        try:
            dur = float(row['minutes'])
            if dur > 0:
                results.append(int(row['result']))
                durations.append(dur)
        except (ValueError, TypeError):
            continue

    if len(durations) == 0:
        psych_data['player_id'].append(int(player_id))
        psych_data['avg_win_duration'].append(None)
        psych_data['avg_loss_duration'].append(None)
        psych_data['fight_score'].append(None)
        psych_data['dominance_score'].append(None)
        continue

    # Separate wins and losses
    win_durations = [durations[i] for i in range(len(results)) if results[i] == 1]
    loss_durations = [durations[i] for i in range(len(results)) if results[i] == 0]

    # Calculate averages
    avg_win = sum(win_durations) / len(win_durations) if len(win_durations) > 0 else None
    avg_loss = sum(loss_durations) / len(loss_durations) if len(loss_durations) > 0 else None

    # Store
    psych_data['player_id'].append(int(player_id))
    psych_data['avg_win_duration'].append(avg_win)
    psych_data['avg_loss_duration'].append(avg_loss)
    psych_data['fight_score'].append(avg_loss)
    psych_data['dominance_score'].append(-avg_win if avg_win is not None else None)

psychology_df = pd.DataFrame(psych_data)

# Merge
df_with_outcomes['player_id'] = df_with_outcomes['player_id'].astype(int)
df_with_outcomes = df_with_outcomes.merge(psychology_df, on='player_id', how='left', suffixes=('', '_psych')).copy()

n_with_data = sum(1 for x in psych_data['fight_score'] if x is not None)
print(f"✅ Added: avg_win_duration, avg_loss_duration, fight_score, dominance_score")

# ============================================================================
# Part 3: Performance Interaction Features
# ============================================================================

print("\n📊 Part 3: Creating Performance Interaction Features...")
print("-"*80)

# Interaction terms
df_with_outcomes['consistency_volume'] = (
    df_with_outcomes['win25_count'] * df_with_outcomes['win_rate_4month']
)

df_with_outcomes['quality_consistency'] = (
    df_with_outcomes['win25_rate'] * df_with_outcomes['win_rate_4month']
)

# Ratio features
df_with_outcomes['win_concentration'] = (
    df_with_outcomes['win25_count'] / (df_with_outcomes['win100_count'] + 1)
)

df_with_outcomes['rate_progression'] = (
    df_with_outcomes['win25_rate'] / (df_with_outcomes['win100_rate'] + 0.01)
)

# Threshold features
df_with_outcomes['elite_indicator'] = (
    (df_with_outcomes['win25_count'] >= 7) & 
    (df_with_outcomes['win_rate_4month'] >= 0.60)
).astype(int)

df_with_outcomes['quality_wins_high'] = (
    df_with_outcomes['win25_count'] >= 8
).astype(int)

# Difference features
df_with_outcomes['rate_stability'] = (
    df_with_outcomes['win25_rate'] - df_with_outcomes['win100_rate']
)

df_with_outcomes['volume_dominance'] = (
    df_with_outcomes['win25_count'] - df_with_outcomes['win100_count']
)

print("✅ Added: 8 performance interaction/ratio features")

# ============================================================================
# Part 4: Sustained Mental Toughness (Volume × Psychology)
# ============================================================================

print("\n📊 Part 4: Volume × Mental Toughness Features...")
print("-"*80)

# Calculate total matches played
df_with_outcomes['total_matches_est'] = (
    df_with_outcomes['win25_count'] / 
    (df_with_outcomes['win25_rate'].replace(0, np.nan))
).fillna(0)

df_with_outcomes['total_matches_est2'] = (
    (df_with_outcomes['win25_count'] + 
     df_with_outcomes['win50_count'] + 
     df_with_outcomes['win100_count']) / 
    df_with_outcomes['win_rate_4month'].replace(0, np.nan)
).fillna(0)

# Volume × Mental Toughness features
df_with_outcomes['sustained_toughness'] = (
    df_with_outcomes['total_matches_est'] * 
    (df_with_outcomes['fight_score'].fillna(0) / 100)
)

df_with_outcomes['sustained_toughness_v2'] = (
    df_with_outcomes['total_matches_est2'] * 
    (df_with_outcomes['fight_score'].fillna(0) / 100)
)

# Triple combo: Volume × Consistency × Mental Toughness
df_with_outcomes['complete_warrior'] = (
    df_with_outcomes['total_matches_est'] * 
    df_with_outcomes['win_rate_4month'] *
    (df_with_outcomes['fight_score'].fillna(0) / 100)
)

# Volume × Dominance
df_with_outcomes['volume_at_right_level'] = (
    df_with_outcomes['total_matches_est'] * 
    (df_with_outcomes['volume_dominance'] / 10)
)

print("✅ Added: sustained_toughness, sustained_toughness_v2, complete_warrior, volume_at_right_level")

# ============================================================================
# Part 5: Baseline Ranking Features
# ============================================================================

print("\n📊 Part 5: Creating Baseline Ranking Features...")
print("-"*80)

final_window_query = f"""
SELECT 
    player_id,
    MAX(rank) as final_window_rank
FROM rankings
WHERE ranking_date BETWEEN 20211201 AND 20211231
  AND player_id IN ({player_ids_str})
GROUP BY player_id
"""
final_window_df = pd.read_sql_query(final_window_query, conn)

# Merge
df_with_outcomes = df_with_outcomes.merge(
    final_window_df,
    on='player_id',
    how='left'
).copy()

df_with_outcomes['final_window_rank'] = df_with_outcomes['final_window_rank'].fillna(2000)

# Basic initial rank features
df_with_outcomes['initial_rank_inverted'] = 2000 / (df_with_outcomes['baseline_rank'] + 1)
df_with_outcomes['ranking_velocity'] = ((df_with_outcomes['baseline_rank'] - df_with_outcomes['final_window_rank'])/4)
df_with_outcomes['gap_to_elite'] = df_with_outcomes['baseline_rank'] - 100
df_with_outcomes['already_competitive'] = (df_with_outcomes['baseline_rank'] < 500).astype(int)

# Performance relative to starting position
df_with_outcomes['performance_vs_baseline'] = (
    df_with_outcomes['win25_count'] / 
    ((df_with_outcomes['baseline_rank'] + 1) / 100)
)

df_with_outcomes['velocity_vs_baseline'] = (
    df_with_outcomes['ranking_velocity'].fillna(0) / 
    ((df_with_outcomes['baseline_rank'] + 1) / 100)
)

df_with_outcomes['warrior_adjusted'] = (
    df_with_outcomes['complete_warrior'] * 
    (1000 / (df_with_outcomes['baseline_rank'] + 1))
)

# ULTIMATE: Breakthrough feasibility
df_with_outcomes['breakthrough_feasibility'] = (
    (1000 / (df_with_outcomes['baseline_rank'] + 1)) *
    df_with_outcomes['complete_warrior'] *
    (df_with_outcomes['ranking_velocity'].fillna(0) / 50 + 1)
)

print("✅ Added initial_rank_inverted, ranking_velocity, gap_to_elite, already_competitive, performance_vs_baseline, breakthrough_feasibility")

# ============================================================================
# Part 6: Momentum/Trajectory Features
# ============================================================================

print("\n📊 Part 5: Building Momentum and Trajectory Features...")
print("-"*80)

# Basic momentum
df_with_outcomes['ranking_momentum'] = (
    df_with_outcomes['baseline_rank'] - df_with_outcomes['final_window_rank']
)

df_with_outcomes['velocity_normalized'] = (
    df_with_outcomes['ranking_velocity'] / (df_with_outcomes['baseline_rank'] + 1)
)

# Rising indicators
df_with_outcomes['is_rising_fast'] = (
    (df_with_outcomes['baseline_rank'] - df_with_outcomes['final_window_rank']) >= 10
).astype(int)

df_with_outcomes['is_surging'] = (
    (df_with_outcomes['baseline_rank'] - df_with_outcomes['final_window_rank']) >= 50
).astype(int)

# Momentum × Performance
df_with_outcomes['momentum_warrior'] = (
    df_with_outcomes['ranking_velocity'] *
    df_with_outcomes['complete_warrior']
)

df_with_outcomes['surge_quality'] = (
    df_with_outcomes['is_surging'] *
    df_with_outcomes['win25_count']
)

print("✅ Added: ranking_momentum, velocity_normalized, momentum_warrior, surge_quality")

# ============================================================================
# Part 7: Advanced Feature Engineering - Multi-Signal Combinations
# ============================================================================

print("\n📊 Part 6: Adding Multi-Signal Combinations...")
print("-"*80)

# ============================================================================
# Group 1: Champion Features (Skill + Mental Toughness)
# ============================================================================

# Combine fight score with quality wins
df_with_outcomes['champion_score'] = (
    df_with_outcomes['win25_count'] * 
    (df_with_outcomes['fight_score'].fillna(0) / 100)
)

# Elite fighter: High volume + high fight score
df_with_outcomes['elite_fighter'] = (
    (df_with_outcomes['win25_count'] >= 7) & 
    (df_with_outcomes['fight_score'] >= df_with_outcomes['fight_score'].median())
).astype(int)

# Mental toughness × consistency
df_with_outcomes['tough_consistency'] = (
    (df_with_outcomes['fight_score'].fillna(0) / 100) * 
    df_with_outcomes['win_rate_4month']
)

# ============================================================================
# Group 2: Tournament Strategy Features
# ============================================================================

# volume_dominance × win_concentration
df_with_outcomes['strategic_dominance'] = (
    df_with_outcomes['volume_dominance'] * 
    df_with_outcomes['win_concentration']
)

# Appropriate level indicator
df_with_outcomes['plays_right_level'] = (
    (df_with_outcomes['volume_dominance'] >= 5) & 
    (df_with_outcomes['win_concentration'] >= 2.0)
).astype(int)

# Tournament efficiency
df_with_outcomes['tournament_efficiency'] = (
    df_with_outcomes['win25_count'] / 
    (df_with_outcomes['win25_count'] + df_with_outcomes['win100_count'] + 1)
)

# ============================================================================
# Group 3: Multi-Signal Combinations
# ============================================================================

# Volume + Concentration + Consistency
df_with_outcomes['breakthrough_composite'] = (
    (df_with_outcomes['volume_dominance'] / 10) +
    (df_with_outcomes['win_concentration'] / 3) +
    (df_with_outcomes['win_rate_4month'])
) / 3

# Elite threshold (multiple gates)
df_with_outcomes['elite_multi_gate'] = (
    (df_with_outcomes['volume_dominance'] >= 5) & 
    (df_with_outcomes['win25_count'] >= 7) &
    (df_with_outcomes['win_rate_4month'] >= 0.55)
).astype(int)

# Peak performance indicator
df_with_outcomes['peak_performer'] = (
    df_with_outcomes['consistency_volume'] * 
    df_with_outcomes['volume_dominance']
)

# ============================================================================
# Group 4: Efficiency & Rate-Based
# ============================================================================

# Win efficiency at appropriate level
df_with_outcomes['level_efficiency'] = (
    df_with_outcomes['win25_rate'] * 
    df_with_outcomes['win_concentration']
)

# Quality density
df_with_outcomes['quality_density'] = (
    df_with_outcomes['win25_count'] / 
    (df_with_outcomes['win25_count'] + df_with_outcomes['win50_count'] + df_with_outcomes['win100_count'] + 1)
)

# Dominance × consistency
df_with_outcomes['dominant_consistency'] = (
    df_with_outcomes['volume_dominance'] * 
    df_with_outcomes['win_rate_4month']
)

print("✅ Added: 12 advanced multi-signal features")

conn.close()


print("\n✅ ALL FEATURE ENGINEERING COMPLETE!")
print(f"Total features: {len(df_with_outcomes.columns)}")

# ============================================================================
# SECTION 3B: Correlation Analysis - Top 15 Features
# ============================================================================

print(f"\n{'='*80}")
print("🔥 CORRELATION ANALYSIS: Top 15 Features by Correlation with Breakthrough")
print("="*80)

# Define feature categories for labeling
original_performance = ['win25_count', 'win25_rate', 'win50_count', 'win50_rate',
                       'win100_count', 'win100_rate', 'win_rate_4month']

engineered_performance = ['consistency_volume', 'quality_consistency', 'win_concentration',
                         'rate_progression', 'elite_indicator', 'quality_wins_high',
                         'rate_stability', 'volume_dominance', 'sustained_toughness',
                         'sustained_toughness_v2', 'complete_warrior', 'volume_at_right_level',
                         'initial_rank_inverted', 'ranking_velocity', 'gap_to_elite',
                         'already_competitive', 'performance_vs_baseline', 'velocity_vs_baseline',
                         'warrior_adjusted', 'breakthrough_feasibility', 'ranking_momentum',
                         'velocity_normalized', 'is_rising_fast', 'is_surging', 'momentum_warrior',
                           'surge_quality', 'champion_score', 'elite_fighter', 'tough_consistency',
                         'strategic_dominance', 'plays_right_level', 'tournament_efficiency',
                         'breakthrough_composite', 'elite_multi_gate', 'peak_performer',
                         'level_efficiency', 'quality_density', 'dominant_consistency']

demographics = ['age_in_2021', 'is_lefty', 'height_cm', 'from_top_nation']

psychology = ['avg_win_duration', 'avg_loss_duration', 'fight_score', 'dominance_score']

all_features = original_performance + engineered_performance + demographics + psychology

# Calculate correlations with breakthrough tier
feature_correlations = df_with_outcomes[all_features].corrwith(
    df_with_outcomes['tier_numeric']
).sort_values(ascending=False)

# Remove NaN correlations
feature_correlations = feature_correlations.dropna()

print("\n📊 TOP 15 FEATURES BY CORRELATION:")
print("-"*80)
top_15 = feature_correlations.head(15)

for i, (feature, corr) in enumerate(top_15.items(), 1):
    # Determine category
    if feature in original_performance:
        category = "📈 Original"
    elif feature in engineered_performance:
        category = "🔧 Engineered"
    elif feature in demographics:
        category = "👤 Demographic"
    elif feature in psychology:
        category = "🧠 Psychology"
    else:
        category = "❓ Other"

    strength = "🔥" if abs(corr) > 0.30 else "📊" if abs(corr) > 0.25 else "💡" if abs(corr) > 0.20 else "  "
    print(f"{i:2d}. {feature:25s}: {corr:+.4f}  {strength} {category}")

print("\n📈 CORRELATION SUMMARY:")
print("-"*80)
print(f"Strongest correlation: {feature_correlations.max():.4f} ({feature_correlations.idxmax()})")
print(f"Features with correlation > 0.30: {(feature_correlations.abs() > 0.30).sum()}")
print(f"Features with correlation > 0.25: {(feature_correlations.abs() > 0.25).sum()}")

# ============================================================================
# SECTION 4: Prepare for Modeling
# ============================================================================

print("\n" + "="*80)
print("📊 SECTION 4: Preparing Data for Modeling")
print("="*80)

# Define candidate features (all numeric features except identifiers and outcomes)
exclude_cols = ['player_id', 'baseline_rank', 'baseline_date', 'best_rank', 
                'best_rank_date', 'tier', 'tier_numeric', 'Birth_Year', 
                'Nationality', 'Dominant_Hand', 'Height']

candidate_features = [col for col in df_with_outcomes.columns 
                     if col not in exclude_cols and 
                     df_with_outcomes[col].dtype in ['int64', 'float64']]

print(f"\n✅ Candidate features: {len(candidate_features)}")

# Prepare X and y
X_cand = df_with_outcomes[candidate_features]
y = df_with_outcomes['tier']

# Train-test split (stratified)
X_cand_train, X_cand_test, y_train, y_test = train_test_split(
    X_cand, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {len(X_cand_train)} players")
print(f"✅ Test set: {len(X_cand_test)} players")

# Check for any remaining NaN values and impute
print("\nHandling missing values...")
for col in X_cand_train.columns:
    if X_cand_train[col].isna().any():
        median_val = X_cand_train[col].median()
        X_cand_train.loc[:, col] = X_cand_train[col].fillna(median_val)
        X_cand_test.loc[:, col] = X_cand_test[col].fillna(median_val)

print("✅ Missing values handled")

# ============================================================================
# SECTION 5: Feature Selection with RFE-CV
# ============================================================================

print("\n" + "="*80)
print("📊 SECTION 5: Recursive Feature Elimination with CV")
print("="*80)

# Temporal cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# ============================================================================
# 5A: Logistic Regression RFE-CV
# ============================================================================

print("\n⏳ Running RFE-CV for Logistic Regression...")

# Scale features for LR
scaler_lr = StandardScaler()
X_train_scaled = scaler_lr.fit_transform(X_cand_train)
X_test_scaled = scaler_lr.transform(X_cand_test)

lr_base = LogisticRegression(
    max_iter=10000,
    class_weight='balanced',
    random_state=42
)

lr_rfecv = RFECV(
    estimator=lr_base,
    step=1,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)

lr_rfecv.fit(X_train_scaled, y_train)
lr_selected_features = X_cand.columns[lr_rfecv.support_].tolist()

print(f"✅ Logistic Regression selected {len(lr_selected_features)} features")
print(f"   Best CV F1-macro: {lr_rfecv.cv_results_['mean_test_score'].max():.4f}")

# ============================================================================
# 5B: Random Forest RFE-CV
# ============================================================================

print("\n⏳ Running RFE-CV for Random Forest...")

rf_base = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_rfecv = RFECV(
    estimator=rf_base,
    step=1,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)

rf_rfecv.fit(X_cand_train, y_train)
rf_selected_features = X_cand.columns[rf_rfecv.support_].tolist()

print(f"✅ Random Forest selected {len(rf_selected_features)} features")
print(f"   Best CV F1-macro: {rf_rfecv.cv_results_['mean_test_score'].max():.4f}")

# ============================================================================
# SECTION 6: Hyperparameter Tuning
# ============================================================================

print("\n" + "="*80)
print("📊 SECTION 6: Hyperparameter Tuning")
print("="*80)

# Prepare feature subsets
X_train_lr = X_train_scaled[:, lr_rfecv.support_]
X_test_lr = X_test_scaled[:, lr_rfecv.support_]
X_train_rf = X_cand_train[rf_selected_features]
X_test_rf = X_cand_test[rf_selected_features]

# Store for later reference
y_train_rfe = y_train
y_test_rfe = y_test

# ============================================================================
# 6A: Logistic Regression Grid Search
# ============================================================================

print("\n⏳ Tuning Logistic Regression...")

lr_param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear'],  # saga supports both L1/L2, liblinear is faster for small datasets
    'max_iter': [1000, 2000, 5000],  # Test convergence at different iterations
    'class_weight': ['balanced']
}

lr_grid = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=lr_param_grid,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
lr_grid.fit(X_train_lr, y_train_rfe)
lr_time = time.time() - start_time

print(f"✅ Completed in {lr_time:.1f} seconds")

print(f"\n🎯 Best Hyperparameters:")
for param, value in lr_grid.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📈 Cross-Validation Performance:")
print(f"   Best CV F1-macro: {lr_grid.best_score_:.4f}")

# ============================================================================
# 6B: Random Forest Grid Search
# ============================================================================

print("\n⏳ Tuning Random Forest...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

rf_grid_full = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
rf_grid_full.fit(X_train_rf, y_train_rfe)
rf_time = time.time() - start_time

print(f"\n✅ Completed in {rf_time/60:.1f} minutes")

print(f"\n🎯 Best Hyperparameters:")
for param, value in rf_grid_full.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📈 Cross-Validation Performance:")
print(f"   Best CV F1-macro: {rf_grid_full.best_score_:.4f}")

# ============================================================================
# SECTION 7: Evaluation
# ============================================================================

print("\n" + "="*80)
print("📊 SECTION 7: MODEL EVALUATION")
print("="*80)

# Get predictions
y_pred_lr = lr_grid.best_estimator_.predict(X_test_lr)
y_pred_rf = rf_grid_full.best_estimator_.predict(X_test_rf)

# Calculate test scores
lr_test_score = f1_score(y_test_rfe, y_pred_lr, average='macro')
rf_test_score = f1_score(y_test_rfe, y_pred_rf, average='macro')

print("\n🏆 FINAL TEST RESULTS")
print("-"*80)
print(f"Logistic Regression:")
print(f"   Test F1-macro: {lr_test_score:.4f}")
print(f"   Features used: {len(lr_selected_features)}")

print(f"\nRandom Forest:")
print(f"   Test F1-macro: {rf_test_score:.4f}")
print(f"   Features used: {len(rf_selected_features)}")

# ============================================================================
# SECTION 8: Feature Importance Analysis
# ============================================================================

print("\n" + "="*80)
print("🔍 MOST IMPORTANT FEATURES")
print("="*80)

# ============================================================================
# 8A: Logistic Regression Feature Importance
# ============================================================================

lr_importance = pd.DataFrame({
    'feature': lr_selected_features,
    'importance': np.linalg.norm(lr_grid.best_estimator_.coef_, axis=0)
}).sort_values('importance', ascending=False).copy()

lr_importance['importance_pct'] = (
    lr_importance['importance'] / lr_importance['importance'].sum() * 100
)

print("\n📊 LOGISTIC REGRESSION - Top Features:")
print("-"*80)
for idx, row in lr_importance.head(15).iterrows():
    print(f"   {row['feature']:40s} {row['importance_pct']:5.2f}%")

# ============================================================================
# 8B: Random Forest Feature Importance
# ============================================================================

rf_importance = pd.DataFrame({
    'feature': rf_selected_features,
    'importance': rf_grid_full.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False).copy()

rf_importance['importance_pct'] = (
    rf_importance['importance'] / rf_importance['importance'].sum() * 100
)

print("\n🌲 RANDOM FOREST - Top Features:")
print("-"*80)
for idx, row in rf_importance.head(15).iterrows():
    print(f"   {row['feature']:40s} {row['importance_pct']:5.2f}%")

# ============================================================================
# SECTION 9: Generate Output Files
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING OUTPUTS")
print("="*80)

class_names = ['No_Progress\n(301+)', 'Challenger\n(201-300)', 
               'Rising\n(101-200)', 'Breakthrough\n(1-100)']

# ============================================================================
# 9A: Confusion Matrices
# ============================================================================

print("\n📊 Generating confusion matrices...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Logistic Regression
cm_lr = confusion_matrix(y_test_rfe, y_pred_lr)
cm_lr_df = pd.DataFrame(cm_lr, index=class_names, columns=class_names)

sns.heatmap(cm_lr_df, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Logistic Regression Confusion Matrix\nTest F1-Macro: {lr_test_score:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Actual Class', fontsize=11)
ax1.set_xlabel('Predicted Class', fontsize=11)

# Random Forest
cm_rf = confusion_matrix(y_test_rfe, y_pred_rf)
cm_rf_df = pd.DataFrame(cm_rf, index=class_names, columns=class_names)

sns.heatmap(cm_rf_df, annot=True, fmt='d', cmap='Oranges', cbar=True, ax=ax2,
            cbar_kws={'label': 'Count'})
ax2.set_title(f'Random Forest Confusion Matrix\nTest F1-Macro: {rf_test_score:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel('Actual Class', fontsize=11)
ax2.set_xlabel('Predicted Class', fontsize=11)

plt.tight_layout()
confusion_path = OUTPUT_DIR / 'confusion_matrices.png'
plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ {confusion_path}")

# ============================================================================
# 9B: Feature Importance Comparison
# ============================================================================

print("\n📊 Generating feature importance comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Logistic Regression
lr_top15 = lr_importance.head(15)
ax1.barh(range(len(lr_top15)), lr_top15['importance_pct'], 
         color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(lr_top15)))
ax1.set_yticklabels(lr_top15['feature'], fontsize=10)
ax1.set_xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
ax1.set_title(f'Logistic Regression: Top 15 Features\n({len(lr_selected_features)} features total)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(lr_top15.iterrows()):
    ax1.text(row['importance_pct'] + 0.3, i, f"{row['importance_pct']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

# Random Forest
rf_top15 = rf_importance.head(15)
ax2.barh(range(len(rf_top15)), rf_top15['importance_pct'], 
         color='coral', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(rf_top15)))
ax2.set_yticklabels(rf_top15['feature'], fontsize=10)
ax2.set_xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
ax2.set_title(f'Random Forest: Top 15 Features\n({len(rf_selected_features)} features total)', 
             fontsize=13, fontweight='bold', pad=15)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(rf_top15.iterrows()):
    ax2.text(row['importance_pct'] + 0.3, i, f"{row['importance_pct']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
importance_path = OUTPUT_DIR / 'feature_importance_comparison.png'
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ {importance_path}")

# ============================================================================
# 9C: Classification Reports
# ============================================================================

print("\n📄 Generating classification reports...")

class_labels = ['No_Progress', 'Challenger', 'Rising', 'Breakthrough']

report_path = OUTPUT_DIR / 'classification_reports.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLASSIFICATION REPORTS - ATP Tennis Breakthrough Prediction\n")
    f.write("="*80 + "\n\n")
    
    f.write("LOGISTIC REGRESSION\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test_rfe, y_pred_lr, target_names=class_labels))
    f.write(f"\nTest F1-Macro: {lr_test_score:.4f}\n")
    f.write(f"Features used: {len(lr_selected_features)}\n\n")
    
    f.write("="*80 + "\n\n")
    
    f.write("RANDOM FOREST\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test_rfe, y_pred_rf, target_names=class_labels))
    f.write(f"\nTest F1-Macro: {rf_test_score:.4f}\n")
    f.write(f"Features used: {len(rf_selected_features)}\n")

print(f"   ✅ {report_path}")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Model Performance Summary:")
print(f"   Logistic Regression F1-macro: {lr_test_score:.4f} ({len(lr_selected_features)} features)")
print(f"   Random Forest F1-macro:       {rf_test_score:.4f} ({len(rf_selected_features)} features)")

print(f"\n💾 Outputs saved to '{OUTPUT_DIR}/':")
print(f"   - confusion_matrices.png")
print(f"   - feature_importance_comparison.png")
print(f"   - classification_reports.txt")

print("\n🎯 Next Steps:")
print("   1. Review feature importance to understand model behavior")
print("   2. Examine confusion matrices for misclassification patterns")
print("   3. Check classification_reports.txt for detailed metrics")

print("\n" + "="*80)
