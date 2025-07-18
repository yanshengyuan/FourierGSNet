import os
import random
import shutil

# Set paths
root_folder = "PhaseGAN_data_4"
test_folder = os.path.join(root_folder, "test")

# Ensure test folder exists
os.makedirs(test_folder, exist_ok=True)

# Set random seed for reproducibility
random_seed = 75
random.seed(random_seed)

# Get list of .h5 files in root folder (excluding the "test" folder)
h5_files = [f for f in os.listdir(root_folder) if f.endswith(".h5")]

# Randomly select 1010 files to move
selected_files = random.sample(h5_files, 1010)

# Move selected files to "test" folder
for file in selected_files:
    src_path = os.path.join(root_folder, file)
    dest_path = os.path.join(test_folder, file)
    shutil.move(src_path, dest_path)
    print(src_path)
    print(dest_path)
    print('\n')

print(f"Moved {len(selected_files)} files to {test_folder}")