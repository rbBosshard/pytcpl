import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

file1 = os.path.join(ROOT_DIR, 'requirements.txt')
file2 = os.path.join(ROOT_DIR, 'manual_requirements.txt')
output_file = file1

with open(file1, 'r') as f1, open(file2, 'r') as f2:
    req1_lines = f1.readlines()
    req2_lines = f2.readlines()

# Merge the contents and remove duplicates
merged_lines = list(set(req1_lines + req2_lines))

# Replace package4==4.0 with package9==9.0
for i, line in enumerate(merged_lines):
    if line.strip() == 'mysql_connector_repackaged':
        merged_lines[i] = 'mysql-connector-python\n'

# Sort the lines
merged_lines.sort()

# Write the merged contents to the output file
with open(output_file, 'w') as out_file:
    out_file.writelines(merged_lines)

print("Done.")






