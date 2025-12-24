import glob
import os
import time

time.sleep(2) # Allow flush

list_of_files = glob.glob('logs/*.log') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Reading: {latest_file}")

with open(latest_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    printing = False
    for line in lines:
        if "fast_ai_prediction" in line or "Prediction failed" in line or "Running portfolio optimization" in line:
            printing = True
            print("-" * 50)
            print(line.strip())
        elif printing and (line.startswith("2025") or line.strip() == ""):
             printing = False # Stop printing if new log entry starts
        elif printing:
             print(line.strip())
