CREATE TABLE Players
	(pid	INTEGER,
	 Name	TEXT,
	 Nationality 	TEXT,
	 Dominant_Hand	TEXT,
	 Height		INTEGER,
	 Birth_Year	INTEGER,
	 PRIMARY KEY(pid)
);

CREATE TABLE Matches
	(pid		INTEGER,
	 oid 		INTEGER,
	 prank		INTEGER,
	 orank		INTEGER,
	 page		INTEGER,
	 oage		INTEGER,
	 Won		INTEGER NOT NULL CHECK(Won IN (0, 1)),
	 Round 		TEXT,
	 Date		INTEGER,
	 Surface	TEXT,
	 Score		TEXT,
	 tid	  	TEXT,
	 Duration	INTEGER,
	 pace		INTEGER,
	 pdf		INTEGER,
	 psvpt		INTEGER,
	 p1stin		INTEGER,
	 p1stwon	INTEGER,
	 p2ndwon	INTEGER,
	 pbpsaved	INTEGER,
         pbpfaced	INTEGER,
	 PRIMARY KEY(pid, oid, tid, Round),
	 FOREIGN KEY(pid) REFERENCES Players(pid),
	 FOREIGN KEY(oid) REFERENCES Players(pid),
	 FOREIGN KEY(tid) REFERENCES Tournaments(tid)
);

CREATE TABLE Tournaments
	(tid	TEXT,
	 name	TEXT,
	 Date	INTEGER,
	 Tour_Level	TEXT,
	 Surface	TEXT,
	 Draw_Size	INTEGER,
	 PRIMARY KEY(tid)
);

CREATE TABLE rankings (
    ranking_date INTEGER,
    player_id INTEGER,
    rank INTEGER,
    points INTEGER,
    PRIMARY KEY (ranking_date, player_id),
    FOREIGN KEY(player_id) REFERENCES Players(pid)
);
