import os

def convert_numbers(line):
    parts = line.split()
    if len(parts) >= 2:
        parts[0] = str(int(parts[0]) - 15)  # Subtract 15 from the first number
        return ' '.join(parts) + '\n'
    return line

folder_path = "D:\Projects\CS-GOObjectDetection\datasets\coco\labels\change_labels"

file_list = os.listdir(folder_path)

for file_name in file_list:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)

    # Check if the item in the folder is a file (not a subfolder)
    if os.path.isfile(file_path):
        # Open and process the file
        with open(file_path, 'r') as infile:
            converted_lines = []
            for line in infile:
                converted_line = convert_numbers(line)
                converted_lines.append(converted_line)
            
            with open(file_path, 'w') as outfile:
                for converted_line in converted_lines:
                    outfile.write(converted_line)