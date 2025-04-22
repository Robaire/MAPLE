# Data Logging

The idea behind this file is to provide a utility to load input_data from an archive of a run so it can be used for testing.
This should also expose a mock_agent set with the starting parameters from the run, since it is not always the same between runs.
Maybe the inverse should be true, you create a mock agent with a specific configuration, and then use the mock_input_data to test the agent.
Each frame some of the functions that the agent provides might change, so there needs to be a manager class that synchronizes the input data and the agent behind the scenes.
This should probably be aware of the size of the data generated so it can stop if need be.

# Two Modules:
## 1. Data Recorder
A way to save data from a simulator run in a standard, archive format (tar.gz)

Starting Data
- [X] Lander Starting Position (in the mock_agent)
- [X] Rover Starting Position (in the mock_agent)
- [X] Sensor Configuration (image size, enabled cameras)
- [X] use_fiducials (in the mock_agent)

Per Frame Data
- [X] Frame Number
- [X] Measured Linear and Angular Velocities
- [X] IMU Data
- [X] Ground Truth Information
- [X] Mission Time
- [X] Current Power
- [X] Radiator Cover Angle
 
Per Frame Per Camera Data
- [X] Grayscale Images (when available)
- [X] Semantic Images (when available)
- [X] Camera Enable
- [X] Camera Position
- [X] Light Intensity
- [X] Light Position

Extra Data
- [ ] Control Inputs (velocity, rotation)

## 2. Playback Module
A generator that takes a path to an archive and returns a configured mock_agent and a utility to either iterate over the input data or jump to specific frames of interest. It will have to update the mock_agent as the frames change

```
Data Format
archive.tar.gz/
    - metadata.toml
    - initial.toml
    - frames.csv
    images/
        <camera>/
            - <camera>_frames.csv
            grayscale/
                - <camera>_grayscale_<frame>.png
            semantic/
                - <camera>_semantic_<frame>.png
```
