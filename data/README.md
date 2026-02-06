# Data Setup Guide

This project uses ATP tennis data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp).

## Quick Start

The modeling notebooks require two data files:

- `tennis_breakthrough.db` - SQLite database with match and ranking data
- `tennis_breakthrough_dataset.csv` - Feature-engineered dataset for modeling

## Full Setup Instructions

### Step 1: Download Source Data

```bash
# Clone Jeff Sackmann's tennis_atp repository
git clone https://github.com/JeffSackmann/tennis_atp.git
```

This provides:

- `atp_matches_2021.csv` - ATP Tour matches
- `atp_matches_qual_chall_2021.csv` - Challenger and qualifying matches  
- `atp_matches_futures_2021.csv` - Futures tour matches
- `atp_rankings_20s.csv` - ATP rankings (2020s)

### Step 2: Build the Database

First load the Schema into a database

```bash
sqlite3 tennis_breakthrough.db < schema.sql
```

Then run the ETL pipeline from the project root:

```bash
# Transform and load match data
python scripts/build_database.py

# Load ranking data
python scripts/load_rankings.py
```

This creates `tennis_breakthrough.db` containing:

- **Players table**: Demographics (name, nationality, height, dominant hand)
- **Tournaments table**: Event details (name, date, level, surface, draw size)
- **Matches table**: Match records split by player perspective
- **Rankings table**: Weekly ATP rankings and points

**Database size**: ~20MB

### Step 3: Generate Feature Dataset

```bash
# Run SQL query to generate initial feature set
sqlite3 tennis_breakthrough.db < scripts/generate_dataset.sql
```

**Output**: `tennis_breakthrough_dataset.csv` with:

- 880 players (filtered for match volume and opponent quality)
- Initial features (win rates, opponent-adjusted metrics)
- Breakthrough target variable

**Note**: Additional features are engineered in train_model.py via SQL queries against the database.

## Configuration Note

The time windows for the project (e.g., breakthrough periods, lookback windows, prediction horizons) are configurable. To adjust these windows, you will need to change the time parameters in all of the relevant scripts where they are defined.
