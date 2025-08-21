#!/bin/bash

# ORB-SLAM Recorder Agent Launch Script
# This script automatically launches the Lunar Simulator and runs the ORB-SLAM recorder agent
# It handles the entire data collection process autonomously

set -e  # Exit on any error

echo "🚀 ORB-SLAM Recorder Agent Launch Script"
echo "========================================"

# Configuration
AGENT_PATH="./agents/orbslam_agent_recorder.py"
SIMULATOR_PATH="./simulator"  # Default simulator path
MISSION_DURATION=300  # 5 minutes in seconds
RECORDING_FREQUENCY=2  # Record every other frame
MAX_DATASET_SIZE=5  # 5GB dataset limit

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if agent file exists
if [ ! -f "$AGENT_PATH" ]; then
    print_error "Agent file not found: $AGENT_PATH"
    print_error "Please ensure the orbslam_agent_recorder.py file exists in the agents directory"
    exit 1
fi

print_status "Agent file found: $AGENT_PATH"

# Check if simulator path exists
if [ ! -d "$SIMULATOR_PATH" ]; then
    print_warning "Default simulator path not found: $SIMULATOR_PATH"
    print_warning "Please provide the path to your Lunar Simulator directory"
    echo ""
    read -p "Enter path to Lunar Simulator directory: " SIMULATOR_PATH
    
    if [ ! -d "$SIMULATOR_PATH" ]; then
        print_error "Invalid simulator path: $SIMULATOR_PATH"
        exit 1
    fi
fi

print_status "Simulator path: $SIMULATOR_PATH"

# Check if simulator binary exists
SIMULATOR_BINARY="$SIMULATOR_PATH/LAC/Binaries/Linux/LAC-Linux-Shipping"
if [ ! -f "$SIMULATOR_BINARY" ]; then
    print_warning "Simulator binary not found at expected path: $SIMULATOR_BINARY"
    print_warning "Checking for alternative paths..."
    
    # Look for simulator binary in common locations
    if [ -f "$SIMULATOR_PATH/LAC-Linux-Shipping" ]; then
        SIMULATOR_BINARY="$SIMULATOR_PATH/LAC-Linux-Shipping"
        print_status "Found simulator binary at: $SIMULATOR_BINARY"
    elif [ -f "$SIMULATOR_PATH/LAC.exe" ]; then
        SIMULATOR_BINARY="$SIMULATOR_PATH/LAC.exe"
        print_status "Found simulator binary at: $SIMULATOR_BINARY"
    else
        print_error "Could not find simulator binary. Please check your simulator installation."
        exit 1
    fi
fi

print_status "Simulator binary: $SIMULATOR_BINARY"

# Check if results directory exists, create if not
RESULTS_DIR="./results"
if [ ! -d "$RESULTS_DIR" ]; then
    print_status "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
fi

# Check if data directory exists, create if not
DATA_DIR="./data"
if [ ! -d "$DATA_DIR" ]; then
    print_status "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# Kill any existing simulator processes
print_status "Checking for existing simulator processes..."
pkill -f "LAC-Linux-Shipping" || true
pkill -f "LAC.exe" || true

print_status "Starting Lunar Simulator..."
print_status "Mission Configuration:"
echo "  - Duration: ${MISSION_DURATION}s (${MISSION_DURATION}/60 minutes)"
echo "  - Recording: Every ${RECORDING_FREQUENCY} frames"
echo "  - Max Dataset: ${MAX_DATASET_SIZE}GB"
echo "  - Agent: $AGENT_PATH"
echo ""

# Start the simulator in the background
print_status "Launching simulator..."
"$SIMULATOR_BINARY" LAC > simulator.log 2>&1 &
SIMULATOR_PID=$!

# Wait for simulator to initialize
print_status "Waiting for simulator to initialize..."
sleep 5

# Check if simulator is running
if ! kill -0 $SIMULATOR_PID 2>/dev/null; then
    print_error "Simulator failed to start. Check simulator.log for details."
    exit 1
fi

print_status "Simulator started successfully (PID: $SIMULATOR_PID)"

# Wait a bit more for full initialization
sleep 3

# Run the agent with the leaderboard
print_status "Starting ORB-SLAM Recorder Agent..."
print_status "The agent will now:"
echo "  1. Initialize ORB-SLAM localization"
echo "  2. Begin straight line navigation"
echo "  3. Record data every ${RECORDING_FREQUENCY} frames"
echo "  4. Automatically stop when dataset reaches ${MAX_DATASET_SIZE}GB"
echo "  5. Complete the mission after ${MISSION_DURATION} seconds"
echo ""

# Run the agent
uv run ./scripts/run_agent.py "$AGENT_PATH" --sim="$SIMULATOR_PATH" --evaluate

# Wait for agent to complete
print_status "Agent execution completed"

# Stop the simulator
print_status "Stopping simulator..."
kill $SIMULATOR_PID 2>/dev/null || true

# Wait for simulator to stop
sleep 2

# Force kill if still running
if kill -0 $SIMULATOR_PID 2>/dev/null; then
    print_warning "Simulator still running, force killing..."
    kill -9 $SIMULATOR_PID 2>/dev/null || true
fi

print_status "Simulator stopped"

# Check for generated data files
print_status "Checking for generated data files..."
if ls ./data/*.lac 1> /dev/null 2>&1; then
    print_status "Data files found:"
    ls -lh ./data/*.lac
else
    print_warning "No .lac data files found in ./data directory"
fi

# Check results
if ls ./results/* 1> /dev/null 2>&1; then
    print_status "Results files found:"
    ls -lh ./results/
else
    print_warning "No results files found in ./results directory"
fi

print_header "Launch script completed successfully!"
print_status "Check the following for results:"
echo "  - Data files: ./data/*.lac"
echo "  - Results: ./results/"
echo "  - Simulator logs: simulator.log"
echo "  - Agent logs: Check the terminal output above"
echo ""

print_status "You can now analyze your collected data using the lac-data playback tools!" 