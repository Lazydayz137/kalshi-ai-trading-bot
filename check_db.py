
import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('trading_system.db')
    
    print("\n--- RECENT POSITIONS ---")
    df = pd.read_sql_query("SELECT id, market_id, side, quantity, entry_price, status, live, timestamp, strategy, decision_id FROM positions ORDER BY timestamp DESC LIMIT 10", conn)
    print(df.to_string())
    
    print("\n\n--- RECENT TRADE LOGS ---")
    df_log = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY exit_timestamp DESC LIMIT 5", conn)
    print(df_log.to_string())

    print("\n\n--- MARKET FRESHNESS CHECK ---")
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM markets")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM markets WHERE last_updated > datetime('now', '-15 minutes')")
    recent = cursor.fetchone()[0]
    
    print(f"Total Markets: {total}")
    print(f"Updated in last 15m: {recent}")
    
    cursor.execute("SELECT count(*) FROM markets WHERE volume > 200")
    vol_200 = cursor.fetchone()[0]
    print(f"Markets with Volume > 200: {vol_200}")

    cursor.execute("SELECT count(*) FROM markets WHERE volume > 50")
    vol_50 = cursor.fetchone()[0]
    print(f"Markets with Volume > 50: {vol_50}")
    
    print("\n--- SCHEMA ---")
    cursor.execute("PRAGMA table_info(markets)")
    for col in cursor.fetchall():
        print(col)

    print("\n--- TOP 10 MARKETS BY VOLUME ---")
    print("\n--- RECENT AI DECISIONS ---")
    print("\n--- RECENT AI DECISIONS (Last 5 Mins) ---")
    try:
        cursor.execute("""
            SELECT count(*) FROM market_analyses 
            WHERE analysis_timestamp > datetime('now', '-5 minutes')
        """)
        count_5m = cursor.fetchone()[0]
        print(f"Total AI Analyses (Last 5m): {count_5m}")
        
        if count_5m > 0:
            cursor.execute("""
                SELECT decision_action, count(*) 
                FROM market_analyses 
                WHERE analysis_timestamp > datetime('now', '-5 minutes')
                GROUP BY decision_action
            """)
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]}")

        # Check market freshness again
        cursor.execute("SELECT count(*) FROM markets WHERE last_updated > datetime('now', '-5 minutes')")
        print(f"Markets Updated (Last 5m): {cursor.fetchone()[0]}")

    except Exception as e:
        print(f"Error reading analyses: {e}")
            
        print("\n--- SAMPLE SKIPPED DECISIONS ---")
        cursor.execute("""
            SELECT market_id, confidence, decision_action
            FROM market_analyses 
            WHERE decision_action = 'SKIP'
            ORDER BY analysis_timestamp DESC 
            LIMIT 5
        """)
        for row in cursor.fetchall():
            print(row)

    except Exception as e:
        print(f"Error reading analyses: {e}")

    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
