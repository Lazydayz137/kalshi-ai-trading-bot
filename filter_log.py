import glob
import os

list_of_files = glob.glob('logs/*.log') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Reading: {latest_file}")

with open(latest_file, 'r', encoding='utf-8') as f:
    for line in f:
        lower_line = line.lower()
        if "warning" in lower_line or "error" in lower_line or "no eligible" in lower_line or "trading_job" in lower_line:
            print(line.strip())
