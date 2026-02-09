import sqlite3
import csv

def load_rankings(filename, db_path):
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()	
	ranking_count = 0

	with open(filename, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			cursor.execute(""" 
			INSERT OR IGNORE INTO rankings (ranking_date, player_id, rank, points)
			VALUES( ?, ?, ?, ?) """, (row['ranking_date'], row['player'], row['rank'], row['points']))			

			if cursor.rowcount > 0:
				ranking_count += 1


	conn.commit()

	cursor.execute("SELECT COUNT(*) FROM rankings")
	total_records = cursor.fetchone()[0]

	cursor.execute("SELECT COUNT(DISTINCT ranking_date) FROM rankings")
	unique_dates = cursor.fetchone()[0]

	cursor.execute("SELECT MIN(ranking_date), MAX(ranking_date) FROM rankings")
	date_range = cursor.fetchone()

	print(f"\n{'='*60}")
	print(f"ETL Pipeline Complete for: {filename}")
	print(f"{'='*60}\n")
	print(f"Total records loaded: {total_records}")
	print(f"Unique ranking dates: {unique_dates}")
	print(f"Date range: {date_range[0]} to {date_range[1]}\n")


	conn.close()


if __name__ == '__main__':
	load_rankings('tennis_atp/atp_rankings_10s.csv', 'tennis_breakthrough.db')
