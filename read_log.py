import glob
import os

list_of_files = glob.glob('logs/*.log') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Reading: {latest_file}")

with open(latest_file, 'r', encoding='utf-8') as f:
    for line in f:
        if "ERROR" in line or "Exception" in line or "Traceback" in line:
            print(line.strip())
