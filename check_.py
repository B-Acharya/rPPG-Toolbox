import re

# File paths
file_with_search_terms = 'result_frame.csv'  # Contains filenames like "94-060_disgust_input0.npy"
file_to_filter = 'processed_list.txt'  # Contains lines like "-rw-r--r-- ... 94-060_disgust_input0.npy"
output_file = 'filtered_output.csv'

import re

# File paths
file_a = 'processed_list.txt'  # Contains lines with file sizes and names
file_b = 'result_frame.csv'  # Contains lines with identifiers and paths
output_file = 'filtered_output.csv'

# Size limit in bytes (10MB)
size_limit = 10 * 1024 * 1024


# Function to convert size to bytes
def size_to_bytes(size_str):
    size_str = size_str.upper().strip()
    if size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024 * 1024)
    elif size_str.endswith('G'):
        return int(float(size_str[:-1]) * 1024 * 1024 * 1024)
    else:
        return int(size_str)  # Assume bytes if no unit


# Step 1: Extract identifiers with size < 10MB from File A
identifiers_to_keep = set()
with open(file_a, 'r') as f:
    for line in f:
        parts = line.split()

        # Check for valid line with size and filename
        if len(parts) > 4:
            size_str = parts[4]  # File size
            filename = parts[-1]  # Filename at the end

            # Convert size to bytes
            try:
                size_in_bytes = size_to_bytes(size_str)
            except ValueError:
                continue  # Skip lines with invalid size format

            # Check size limit
            if size_in_bytes < size_limit:
                # Extract identifier from filename
                identifier = filename.split('_')[0]  # Get identifier like 'pre-81-044'
                identifiers_to_keep.add(identifier)

# Step 2: Filter lines in File B using extracted identifiers
filtered_entries = []
with open(file_b, 'r') as f:
    for line in f:
        # Check if any identifier is in the line
        if any(identifier in line for identifier in identifiers_to_keep):
            filtered_entries.append(line)

# Write to the new file
with open(output_file, 'w') as f:
    f.writelines(filtered_entries)

print("Filtered file created:", output_file)
