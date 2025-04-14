#!/bin/bash

# This is a bash script to evaluate battery charging code

# Configuration
MAX_RUNTIME=36000  # Maximum runtime in seconds (1 hour)
COORDINATES_FILE="coordinates.txt"  # File containing x,y coordinates
RESULTS_FILE="charging_results2.csv"  # File to store results
TIMEOUT_PER_RUN=720  # Timeout for each simulator run (seconds)

# Create or clear results file and add header
echo "x,y,status,timestamp" > "$RESULTS_FILE"

# Check if coordinates file exists
if [ ! -f "$COORDINATES_FILE" ]; then
    echo "Coordinates file not found. Creating sample file..."
    cat > "$COORDINATES_FILE" << EOF
-0.11222237348556519,-3
0.5,-2.5
1.2,-3.1
-0.8,-2.8
EOF
    echo "Sample coordinates file created. Please modify as needed."
fi

# Start timestamp
START_TIME=$(date +%s)
TOTAL_RUNS=0
SUCCESSFUL_CHARGES=0

# Function to process a single coordinate
run_simulator() {
    local x=$1
    local y=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "Running simulator with coordinates [$x, $y]"
    
    # Run the simulator with timeout and capture output
    output=$(timeout $TIMEOUT_PER_RUN uv run ./scripts/run_agent.py agents/simple_charge.py --sim="./simulator" --xy="[$x, $y]" 2>&1)

    # Check if charging was successful
    if echo "$output" | grep -q "CHARGING NOTIFICATION FROM SIMULATOR"; then
        echo "SUCCESS: Charging completed for coordinates [$x, $y]"
        echo "$x,$y,SUCCESS,$timestamp" >> "$RESULTS_FILE"
        ((SUCCESSFUL_CHARGES++))
    else
        echo "FAILURE: Charging failed for coordinates [$x, $y]"
        echo "$x,$y,FAILURE,$timestamp" >> "$RESULTS_FILE"
    fi
    
    ((TOTAL_RUNS++))
}

echo "Starting charging simulator runs..."
echo "This script will run for a maximum of $MAX_RUNTIME seconds"

# Main loop
CURRENT_TIME=$(date +%s)
ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

# Check if we've exceeded the maximum runtime
if [ $ELAPSED_TIME -ge $MAX_RUNTIME ]; then
    echo "Maximum runtime reached. Terminating..."
    break
fi

# Process each coordinate from the file
while IFS=, read -r x y || [ -n "$x" ]; do
    # Check time again before each run
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -ge $MAX_RUNTIME ]; then
        echo "Maximum runtime reached during coordinates processing. Terminating..."
        break 2
    fi
    
    run_simulator $x $y
done < "$COORDINATES_FILE"

# Calculate and display results
SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", ($SUCCESSFUL_CHARGES / $TOTAL_RUNS) * 100}")

echo ""
echo "===== SIMULATION RESULTS ====="
echo "Total runs: $TOTAL_RUNS"
echo "Successful charges: $SUCCESSFUL_CHARGES"
echo "Success rate: $SUCCESS_RATE%"
echo "Results saved to $RESULTS_FILE"
echo "============================"

exit 0