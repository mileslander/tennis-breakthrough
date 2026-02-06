.headers on
.mode csv
.output tennis_breakthrough_dataset.csv

-- Step 1: Get eligible players (challengers/futures only, active competitors)
WITH target_players AS (
    SELECT DISTINCT m.pid as player_id
    FROM Matches m
    WHERE m.Date BETWEEN 20210901 AND 20211231
      AND m.Score NOT LIKE '%RET%'
      AND m.Score NOT LIKE '%W/O%'
      AND m.Score IS NOT NULL
      -- Exclude established players (already top 150 during collection)
      AND m.pid NOT IN (
          SELECT DISTINCT player_id
          FROM rankings
          WHERE ranking_date BETWEEN 20210901 AND 20211231
            AND rank <= 150
      )
      -- Require minimum activity
      AND m.pid IN (
          SELECT pid
          FROM Matches
          WHERE Date BETWEEN 20210901 AND 20211231
          GROUP BY pid
          HAVING COUNT(*) >= 5
      )
),

-- Step 2: Extract opponent ranking features
opponent_features AS (
    SELECT
        pid as player_id,
        SUM(CASE WHEN Won = 1 AND prank - orank > 24 THEN 1 ELSE 0 END) as win25_count,
        SUM(CASE WHEN prank - orank > 24 THEN 1 ELSE 0 END) as attempt25,
        SUM(CASE WHEN Won = 1 AND prank - orank > 49 THEN 1 ELSE 0 END) as win50_count, 
        SUM(CASE WHEN prank - orank > 49 THEN 1 ELSE 0 END) as attempt50,
        SUM(CASE WHEN Won = 1 AND prank - orank > 99 THEN 1 ELSE 0 END) as win100_count, 
        SUM(CASE WHEN prank - orank > 99 THEN 1 ELSE 0 END) as attempt100,
        CASE WHEN SUM(CASE WHEN prank - orank > 24 THEN 1 ELSE 0 END) > 0
             THEN CAST(SUM(CASE WHEN Won = 1 AND prank - orank > 24 THEN 1 ELSE 0 END) AS REAL) / 
                  SUM(CASE WHEN prank - orank > 24 THEN 1 ELSE 0 END)
             ELSE 0 END as win25_rate,
        CASE WHEN SUM(CASE WHEN prank - orank > 49 THEN 1 ELSE 0 END) > 0
             THEN CAST(SUM(CASE WHEN Won = 1 AND prank - orank > 49 THEN 1 ELSE 0 END) AS REAL) / 
                  SUM(CASE WHEN prank - orank > 49 THEN 1 ELSE 0 END)
             ELSE 0 END as win50_rate,
        CASE WHEN SUM(CASE WHEN prank - orank > 99 THEN 1 ELSE 0 END) > 0
             THEN CAST(SUM(CASE WHEN Won = 1 AND prank - orank > 99 THEN 1 ELSE 0 END) AS REAL) / 
                  SUM(CASE WHEN prank - orank > 99 THEN 1 ELSE 0 END)
             ELSE 0 END as win100_rate
    FROM Matches
    WHERE Date BETWEEN 20210901 AND 20211231
      AND pid IN (SELECT player_id FROM target_players)
    GROUP BY pid
),

-- Step 3: Extract rolling win rate features
rolling_features AS (
    SELECT 
        pid as player_id,
        COUNT(*) as match_count,
        SUM(CASE WHEN Won = 1 THEN 1 ELSE 0 END) as matches_won,
        CASE WHEN COUNT(*) > 0
             THEN CAST(SUM(CASE WHEN Won = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
             ELSE 0 END as win_rate_4month
    FROM matches
    WHERE Date BETWEEN 20210901 AND 20211231
      AND Score NOT LIKE '%RET%' 
      AND Score NOT LIKE '%W/O%' 
      AND Score IS NOT NULL
      AND pid IN (SELECT player_id FROM target_players)
    GROUP BY pid
),

breakthrough_players AS (
    SELECT DISTINCT pid as player_id
    FROM Matches
    WHERE Date BETWEEN 20220101 AND 20221231 AND prank <= 100
),

-- Step 4: Label each target player as a breakthrough or not
target_variable AS (
    SELECT
        DISTINCT player_id,
        CASE WHEN player_id IN (SELECT player_id FROM breakthrough_players) THEN 1 ELSE 0 END as breakthrough
    FROM target_players
)

-- Step 5: Join everything together!
SELECT 
    t.player_id,
    o.win25_count,
    o.win25_rate,
    o.win50_count,
    o.win50_rate,
    o.win100_count,
    o.win100_rate,
    r.win_rate_4month,
    r.match_count,
    t.breakthrough
FROM target_variable t
JOIN opponent_features o ON t.player_id = o.player_id
JOIN rolling_features r ON t.player_id = r.player_id
ORDER BY t.player_id;

.output stdout
