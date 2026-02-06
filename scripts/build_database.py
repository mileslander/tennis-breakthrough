import csv
import sqlite3

def extract_unique_players(filename):
    """Extract all unique players from CSV using dict for automatic deduplication"""
    players = {}
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Add winner (dict key ensures uniqueness)
            if row['winner_id'] and row['winner_id'] not in players:
                players[row['winner_id']] = {
                    'pid': row['winner_id'],
                    'name': row['winner_name'],
                    'nationality': row['winner_ioc'],
                    'hand': row['winner_hand'],
                    'height': row['winner_ht'] if row['winner_ht'] else None
                }
            
            # Add loser
            if row['loser_id'] and row['loser_id'] not in players:
                players[row['loser_id']] = {
                    'pid': row['loser_id'],
                    'name': row['loser_name'],
                    'nationality': row['loser_ioc'],
                    'hand': row['loser_hand'], 
                    'height': row['loser_ht'] if row['loser_ht'] else None
                }
    
    print(f"Found {len(players)} unique players")
    return list(players.values())

def extract_unique_tournaments(filename):
    """Extract all unique tournaments from CSV"""
    tournaments = {}
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['tourney_id'] and row['tourney_id'] not in tournaments:
                tournaments[row['tourney_id']] = {
                    'tid': row['tourney_id'],
                    'name': row['tourney_name'],
                    'date': row['tourney_date'],
                    'level': row['tourney_level'],
                    'surface': row['surface'],
                    'draw_size': row['draw_size'] if row['draw_size'] else None
                }
    
    print(f"Found {len(tournaments)} unique tournaments")
    return list(tournaments.values())

def load_players(players):
    """Load players into database"""
    conn = sqlite3.connect('tennis_breakthrough.db')
    cursor = conn.cursor()
    
    for player in players:
        cursor.execute("""
        INSERT OR IGNORE INTO Players (pid, Name, Nationality, Dominant_Hand, Height)
        VALUES (?, ?, ?, ?, ?)
        """, (player['pid'], player['name'], player['nationality'], 
              player['hand'], player['height']))
    
    conn.commit()
    conn.close()
    print(f"Loaded {len(players)} players")

def load_tournaments(tournaments):
    """Load tournaments into database"""
    conn = sqlite3.connect('tennis_breakthrough.db')
    cursor = conn.cursor()
    
    for i, tournament in enumerate(tournaments):
        try:
            # Convert draw_size to int if it exists
            draw_size = None
            if tournament['draw_size']:
                draw_size = int(tournament['draw_size'])
            
            cursor.execute("""
            INSERT OR IGNORE INTO Tournaments (tid, Name, Date, Tour_Level, Surface, Draw_Size)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (tournament['tid'], tournament['name'], tournament['date'],
                  tournament['level'], tournament['surface'], draw_size))
        
        except Exception as e:
            print(f"Error loading tournament {i}: {tournament}")
            print(f"Error details: {e}")
            break  # Stop on first error to see what's wrong
    
    conn.commit()
    conn.close()
    print(f"Loaded tournaments successfully")

def load_matches(filename):
    """Load matches using dual-row approach"""
    conn = sqlite3.connect('tennis_breakthrough.db')
    cursor = conn.cursor()
    
    matches_loaded = 0
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Winner perspective
            cursor.execute("""
            INSERT OR IGNORE INTO Matches (pid, oid, prank, orank, page, oage, Won, Round, Date, Surface, Score, tid, Duration, pace, pdf, psvpt, p1stin, p1stwon, p2ndwon, pbpsaved, pbpfaced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['winner_id'], row['loser_id'], 
                  row['winner_rank'], row['loser_rank'],
                  row['winner_age'], row['loser_age'],
                  1, row['round'], row['tourney_date'], row['surface'], 
                  row['score'], row['tourney_id'], row['minutes'],
                  row['w_ace'], row['w_df'], row['w_svpt'],
                  row['w_1stIn'], row['w_1stWon'], row['w_2ndWon'],
                  row['w_bpSaved'], row['w_bpFaced']))
            
            # Loser perspective  
            cursor.execute("""
            INSERT OR IGNORE INTO Matches (pid, oid, prank, orank, page, oage, Won, Round, Date, Surface, Score, tid, Duration, pace, pdf, psvpt, p1stin, p1stwon, p2ndwon, pbpsaved, pbpfaced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['loser_id'], row['winner_id'],
                  row['loser_rank'], row['winner_rank'], 
                  row['loser_age'], row['winner_age'],
                  0, row['round'], row['tourney_date'], row['surface'],
                  row['score'], row['tourney_id'], row['minutes'],
                  row['l_ace'], row['l_df'], row['l_svpt'],
                  row['l_1stIn'], row['l_1stWon'], row['l_2ndWon'],
                  row['l_bpSaved'], row['l_bpFaced']))
            
            matches_loaded += 2  # We insert 2 rows per match
    
    conn.commit()
    conn.close()
    print(f"Loaded {matches_loaded} match records ({matches_loaded//2} matches)")

def run_full_etl(filename):
    print(f"\n{'='*60}")
    print(f"Starting ETL Pipeline for: {filename}")
    print(f"{'='*60}\n")
    
    # Pass 1: Extract and load players
    players = extract_unique_players(filename)
    load_players(players)
    
    # Pass 2: Extract and load tournaments  
    tournaments = extract_unique_tournaments(filename)
    load_tournaments(tournaments)
    
    # Pass 3: Load matches
    load_matches(filename)
    
    print(f"\n{'='*60}")
    print(f"ETL Pipeline Complete for: {filename}")


    
if __name__ == '__main__':
    run_full_etl('tennis_atp/atp_matches_qual_chall_2021.csv')
    run_full_etl('tennis_atp/atp_matches_futures_2021.csv')
    run_full_etl('tennis_atp/atp_matches_2021.csv') 
