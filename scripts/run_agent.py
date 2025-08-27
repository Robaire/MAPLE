# /// script
# requires-python = ">=3.10"
# dependencies = [
#  "leaderboard",
#  "maple",
#  "psutil",
# ]
# [tool.uv.sources]
# leaderboard = { path = "../leaderboard" }
# maple = {path = "../", editable = true }
# ///

import argparse
import os
import subprocess
import psutil
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Agent Runner", description="Runs a Lunar Autonomy Challenge Agent"
    )

    parser.add_argument("agent", help="Path to the agent file", type=str)
    parser.add_argument(
        "-s",
        "--sim",
        help="Path to the Lunar Simulator directory",
        type=str,
        dest="sim_path",
        default="./simulator",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        help="Set evaluation mode",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--qualifier",
        help="Set qualifier mode",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--development",
        help="Set development mode",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--mission-id",
        help="Mission ID to use (0-9, corresponding to presets 1-10 in missions_training.xml)",
        type=int,
        dest="mission_id",
        default=0,
        choices=range(10),  # 0-9
    )

    args = parser.parse_args()

    # Assert that -d and -q are not both set
    if args.qualifier and args.development:
        print("Cannot set both -q and -d")
        exit()

    # Check that the agent file exists before trying to run anything
    if not os.path.exists(args.agent):
        print(f"No file found at: {args.agent}")
        exit()

    # Display mission information
    print(f"🚀 Launching agent with Mission ID: {args.mission_id}")
    print(f"🗺️  This will load Moon_Map_01 with Preset {args.mission_id + 1}")
    print(f"📁 Agent file: {args.agent}")
    print(f"🔧 Simulator path: {args.sim_path}")
    print("")

    # Set environment variable for agents to access mission ID
    preset_number = args.mission_id + 1
    os.environ['LAC_MISSION_ID'] = str(args.mission_id)
    os.environ['LAC_PRESET_NUMBER'] = str(preset_number)
    print(f"🔧 Set environment variables: LAC_MISSION_ID={args.mission_id}, LAC_PRESET_NUMBER={preset_number}")
    print()

    # Check if the simulator is already running (probably hanging) and kill it
    for proc in psutil.process_iter(attrs=["name"]):
        try:
            if proc.info["name"] == "LAC-Linux-Shipping":
                proc.kill()
        except:
            pass

    # Run the simulator
    sim_path = (
        os.path.expanduser(args.sim_path) + "/LAC/Binaries/Linux/LAC-Linux-Shipping"
    )

    simulator = subprocess.Popen(
        [sim_path, "LAC"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for the simulator to get setup
    time.sleep(3)

    # Make sure a results folder exists otherwise the leaderboard will crash
    try:
        os.mkdir("./results")
    except FileExistsError:
        pass

    # Run the leaderboard
    leaderboard_path = "./leaderboard/leaderboard_evaluator.py"
    leaderboard_args = {
        "missions": "./leaderboard/data/missions_training.xml",
        "missions-subset": args.mission_id,  # Use the mission ID parameter
        "seed": 0,
        "repetitions": 1,
        "checkpoint": "./results",
        "agent": args.agent,
        "agent-config": "",
        "record": 1,
        "record-control": 1,
        "resume": "",
        "qualifier": "",
        "evaluation": "",
        "development": "",
    }

    # Set leaderboard mode
    if args.qualifier:
        leaderboard_args["qualifier"] = 1
    elif args.development:
        leaderboard_args["development"] = 1
    else:
        leaderboard_args["evaluation"] = 1

    leaderboard = subprocess.run(
        ["python", leaderboard_path]
        + [f"--{key}={value}" for key, value in leaderboard_args.items()],
    )

    # Stop the simulator
    try:
        simulator.terminate()
        simulator.wait()
    except:
        pass
