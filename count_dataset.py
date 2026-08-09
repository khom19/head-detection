import os
import glob

folder_path = './data/merge_train_data'

txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

total_objects = 0
total_files = 0

for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        valid_lines = [line for line in lines if line.strip()]
        total_objects += len(valid_lines)
        total_files += 1

print(f"total .txt files: {total_files} ไฟล์")
print(f"total objects: {total_objects} objects")