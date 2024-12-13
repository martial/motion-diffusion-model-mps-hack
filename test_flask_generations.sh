#!/bin/bash

# Base URL for the API
API_URL="http://localhost:6000"

# Base prompt to use for all variations
BASE_PROMPT="the person walks forward"

# Create output directory for logs
mkdir -p test_logs

# Function to make a request and save results
make_request() {
    local name=$1
    local data=$2
    echo "Making request: $name"
    
    # Make the request and save both the response and timing
    time curl -X POST \
        "$API_URL/generate" \
        -H 'Content-Type: application/json' \
        -d "$data" \
        > "test_logs/${name}_response.json" 2> "test_logs/${name}_error.log"
    
    echo "Completed: $name"
    echo "----------------------------------------"
}

# Test 1: Baseline
echo "Running baseline test..."
make_request "baseline" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "output_format": ["mp4", "npy"]
}'

# Test 2: Higher guidance
echo "Running high guidance test..."
make_request "high_guidance" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 4.0,
    "output_format": ["mp4", "npy"]
}'

# Test 3: Lower guidance
echo "Running low guidance test..."
make_request "low_guidance" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 1.5,
    "output_format": ["mp4", "npy"]
}'

# Test 4: Longer motion
echo "Running longer motion test..."
make_request "long_motion" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 10.0,
    "guidance_param": 2.5,
    "output_format": ["mp4", "npy"]
}'

# Test 5: Temperature variation
echo "Running temperature variation test..."
make_request "temperature_var" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "temperature": 0.8,
    "output_format": ["mp4", "npy"]
}'

# Test 6: Multiple repetitions
echo "Running multiple repetitions test..."
make_request "multi_rep" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 3,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "output_format": ["mp4", "npy"]
}'

# Test 7: Different seed
echo "Running different seed test..."
make_request "diff_seed" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 100,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "output_format": ["mp4", "npy"]
}'

# Test 8: Higher FPS
echo "Running high FPS test..."
make_request "high_fps" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "fps": 30,
    "output_format": ["mp4", "npy"]
}'

# Test 9: All formats
echo "Running all formats test..."
make_request "all_formats" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "output_format": ["mp4", "npy", "json", "bvh"]
}'

# Test 10: Advanced parameters
echo "Running advanced parameters test..."
make_request "advanced" '{
    "prompt": "'"$BASE_PROMPT"'",
    "seed": 42,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "temperature": 0.8,
    "skip_timesteps": 10,
    "const_noise": true,
    "save_intermediates": true,
    "output_format": ["mp4", "npy"]
}'

echo "All tests completed. Check test_logs directory for results."

# Optional: Create a summary of all tests
echo "Creating summary..."
echo "Test Summary" > test_logs/summary.txt
echo "----------------------------------------" >> test_logs/summary.txt
for test in baseline high_guidance low_guidance long_motion temperature_var multi_rep diff_seed high_fps all_formats advanced; do
    echo "$test:" >> test_logs/summary.txt
    if [ -f "test_logs/${test}_error.log" ]; then
        echo "  Errors:" >> test_logs/summary.txt
        cat "test_logs/${test}_error.log" >> test_logs/summary.txt
    fi
    echo "----------------------------------------" >> test_logs/summary.txt
done

echo "Summary created at test_logs/summary.txt"