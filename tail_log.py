import glob
import os
import collections

list_of_files = glob.glob('logs/*.log') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Reading: {latest_file}")

with open(latest_file, 'r', encoding='utf-8') as f:
    lines = collections.deque(f, maxlen=50)
    for line in lines:
        print(line, end='')
