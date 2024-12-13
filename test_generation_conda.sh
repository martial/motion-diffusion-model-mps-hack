#!/bin/bash

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Check if mdm environment exists and activate it
if ! conda env list | grep -q "^mdm "; then
    echo "MDM environment not found. Please run bash install_mdm.sh first to install the environment."
    exit 1
else
    echo "Activating MDM environment..."
    conda activate mdm
fi

mkdir -p combined_motions_discovery

# Array of prompts for body part discovery and exploration
prompts=(
    "a person slowly wiggling their fingers one by one, discovering their hand movement"
)

# Generate motions for each prompt
for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    echo "Generating motion for body discovery: $prompt"
    
    python -m sample.generate \
        --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
        --text "$prompt" \
        --motion_length 96.0 \
        --seed $((42 + i))
        
    # Get the latest generated sample folder
    latest_sample=$(ls -td save/humanml_enc_512_50steps/samples_* | head -1)
    
    # Copy the video to our combined_motions folder with a numbered prefix
    cp "$latest_sample/samples_00_to_00.mp4" "combined_motions_discovery/$(printf "%02d" $i)_motion.mp4"
done

# Combine videos using ffmpeg
cd combined_motions_discovery
ls -v *.mp4 | sed 's/^/file /' > videos.txt
ffmpeg -f concat -safe 0 -i videos.txt -c copy final_combined_discovery.mp4
rm videos.txt

echo "Generated body discovery and exploration sequence"
