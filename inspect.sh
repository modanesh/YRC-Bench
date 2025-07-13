#!/bin/bash

# Check if the user has provided a main folder name
if [ -z "$1" ]; then
  echo "Usage: $0 <main_folder_name>"
  exit 1
fi

# Define the main folder path relative to the logs directory
main_folder="logs/$1"

# Check if the specified main folder exists
if [ ! -d "$main_folder" ]; then
  echo "The specified main folder does not exist: $main_folder"
  exit 1
fi

# Navigate to the main folder
cd "$main_folder"

# Initialize an empty array to hold subfolders with .gif files
gif_folders=()

# Loop through each subfolder inside the main folder
for subfolder in */; do
  cd "$subfolder"  # Enter the subfolder
  
  # List the contents of the subfolder
  ls

  # Check if there are any .gif files in this subfolder
  if ls *.gif 1> /dev/null 2>&1; then
    # If .gif files are found, store the subfolder name
    gif_folders+=("$subfolder")
  fi
  
  # Go back to the main folder
  cd ..
done


# Check if any subfolders contained .gif files and list them
if [ ${#gif_folders[@]} -gt 0 ]; then
  echo "Subfolders containing .gif files:"
  for folder in "${gif_folders[@]}"; do
    echo "$folder"
  done
else
  echo "No subfolders with .gif files found."
fi

