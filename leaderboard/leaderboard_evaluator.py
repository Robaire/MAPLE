#!/usr/bin/env python

# Copyright (c) 2024 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""

import traceback
import argparse
import sys

import carla

from leaderboard.missionmanager.mission_behaviors import MissionBehaviors
from leaderboard.missionmanager.mission_manager import MissionManager
from leaderboard.missionmanager.mission_spawner import MissionSpawner
from leaderboard.missionmanager.mission_logger import MissionLogger
from leaderboard.missionmanager.mission_weather import MissionWeather
from leaderboard.agents.agent_wrapper import AgentWrapper, AgentSetupError, AgentRuntimeError, SensorConfigurationError
from leaderboard.statistics.mission_data import MissionData
from leaderboard.statistics.result_writer import ResultOutputProvider
from leaderboard.statistics.statistics_manager import StatisticsManager, STATUS_MESSAGES
from leaderboard.utils.mission_indexer import MissionIndexer
from leaderboard.utils.constants import Constants


class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the mission.
    """
    def __init__(self):
        """
        Initialize
        """
        self._client = None
        self._world = None

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client and setting up the world 
        """
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 0.05,
            deterministic_ragdolls = True,
            spectator_as_ego = False
        )
        client.get_world().apply_settings(settings)

        return client

    def _reset_simulation(self):
        """
        Changes the modified world settings back to asynchronous
        """
        try:
            if self._world:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
        except RuntimeError:
            # If the simulation breaks, just ignore this
            pass

    def _create_agent_wrapper(self, *args, **kwargs):
        return AgentWrapper(*args, **kwargs)

    def _create_spawner(self, *args, **kwargs):
        return MissionSpawner(*args, *kwargs)

    def _create_logger(self, *args, **kwargs):
        return MissionLogger(*args, **kwargs)

    def _create_behaviors(self, *args, **kwargs):
        return MissionBehaviors(*args, **kwargs)

    def _create_weather(self, *args, **kwargs):
        return MissionWeather(*args, **kwargs)

    def _create_manager(self, *args, **kwargs):
        return MissionManager(*args, **kwargs)

    def _load_and_run_mission(self, args, config, constants):
        """
        Load and run the mission given by config.

        Depending on what code fails, the simulation will either stop the
        mission and continue from the next one, or report a crash and stop.
        """
        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.base_name, config.rep_index))

        manager = None
        agent_wrapper = None
        spawner = None
        logger = None
        weather = None
        behaviors = None

        try:
            print("\033[1m> Setting up the world\033[0m")
            self._world = self._client.load_world(config.map, reset_settings=False)

            # These are created in separate functions to help with the profiling
            agent_wrapper = self._create_agent_wrapper(self._world, args.agent, args.agent_config, args.evaluation)
            spawner = self._create_spawner(self._world, config, agent_wrapper.agent_fiducials)
            logger = self._create_logger(self._client, args.record, args.record_control, args.checkpoint, config.name)
            weather = self._create_weather(self._world)
            behaviors = self._create_behaviors(self._world, agent_wrapper.agent_fiducials, constants)
            manager = self._create_manager(args.timeout)

            print("\033[1m> Setting up the mission\033[0m")
            spawner.setup(args.seed)
            behaviors.setup(spawner.ego_vehicle, spawner.lander)

            print("\033[1m> Setting up the agent\033[0m")
            agent_wrapper.setup(spawner.ego_vehicle, spawner.lander, constants)

            print("\033[1m> Running the mission\033[0m")
            manager.setup(self._world, spawner.ego_vehicle, agent_wrapper, weather, behaviors, logger)
            logger.start()
            manager.run()

            print("\033[1m> Stopping the mission\033[0m")
            logger.stop()
            agent_wrapper.stop()
            manager.stop(constants, args.development)

        except SensorConfigurationError as e:
            print(f"\n\033[91mCould not set up the agent sensors")
            print(f"\n{traceback.format_exc()}\033[0m")
            mission_data = MissionData(config, *STATUS_MESSAGES["Invalid_sensors"],
                                       agent_wrapper, manager, behaviors, False)

        except AgentSetupError:
            print(f"\n\033[91mCould not set up the required agent")
            print(f"\n{traceback.format_exc()}\033[0m")
            mission_data = MissionData(config, *STATUS_MESSAGES["Agent_setup_failure"],
                                       agent_wrapper, manager, behaviors, False)

        except AgentRuntimeError:
            print(f"\n\033[91mStopping the mission, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")
            mission_data = MissionData(config, *STATUS_MESSAGES["Agent_runtime_failure"],
                                       agent_wrapper, manager, behaviors, False)

        except Exception as e:
            print(f"\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")
            mission_data = MissionData(config, *STATUS_MESSAGES["Simulation_crash"],
                                       agent_wrapper, manager, behaviors, False)

        else:
            mission_data = MissionData(config, *STATUS_MESSAGES["Finished_mission"],
                                       agent_wrapper, manager, behaviors, True)

        # Clean up the simulation in all cases
        try:
            print("\033[1m> Cleaning up the simulation\033[0m")
            if agent_wrapper: agent_wrapper.cleanup()
            if spawner: spawner.cleanup()
            if weather: weather.cleanup()
            if behaviors: behaviors.cleanup()
            if manager: manager.cleanup()

        except Exception:
            print("\n\033[91mFailed to cleanup the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")

        return mission_data

    def run(self, args):
        """
        Run the challenge mode
        """
        self._client = self._setup_simulation(args)
        constants = Constants(args.qualifier, args.testing)

        try:
            mission_indexer = MissionIndexer(args.missions, args.repetitions, args.missions_subset)
        except Exception as e:
            print(f"\n\033[91mError during the configuration parsing:")
            print(f"\n{traceback.format_exc()}\033[0m")
            return True

        if not args.development:
            statistics_manager = StatisticsManager(mission_indexer.total, args.checkpoint, constants)

            if args.resume:
                resume = mission_indexer.validate_and_resume(statistics_manager.get_results_endpoint())
                if resume:
                    statistics_manager.add_file_records()
            else:
                resume = False

            statistics_manager.set_progress(mission_indexer.index)
            statistics_manager.save_statistics()

        crashed = False
        while mission_indexer.peek() and not crashed:

            config = mission_indexer.get_next_config()
            if not args.development:
                statistics_manager.create_mission_record(config.name, config.index)

            mission_data = self._load_and_run_mission(args, config, constants)
            crashed = mission_data.crashed

            try:
                if not args.development and not args.testing:
                    print("\033[1m> Registering the mission statistics\033[0m")
                    statistics_manager.set_progress(mission_indexer.index)
                    mission_record = statistics_manager.compute_mission_statistics(mission_data)
                    statistics_manager.save_statistics()
                    if mission_data.show_results:
                        ResultOutputProvider(mission_data, mission_record)
                else:
                    mode = "Testing" if args.testing else "Development"
                    print(f"\033[1m> {mode} mode: Skipping the mission statistics\033[0m")

            except Exception as e:
                print(f"\n\033[91mError while computing the statistics, they might be empty:")
                print(f"\n{traceback.format_exc()}\033[0m")
                crashed = True

        if not crashed:
            try:
                if not args.development and not args.testing:
                    print("\033[1m> Registering the global statistics\033[0m")
                    statistics_manager.compute_global_statistics()
                    statistics_manager.validate_and_save_statistics()
                else:
                    mode = "Testing" if args.testing else "Development"
                    print(f"\033[1m> {mode} mode: Skipping the global statistics\033[0m")
            except Exception as e:
                print(f"\n\033[91mError while computing the statistics, they might be empty:")
                print(f"\n{traceback.format_exc()}\033[0m")
        else:
            if not args.development:
                statistics_manager.set_entry_status('Crashed')
                statistics_manager.save_statistics()

        self._reset_simulation()

        return crashed

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default=30.0, type=float,
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--seed', default=0, type=int,
                        help='Set the random seed of the simulation')

    # simulation setup
    parser.add_argument('--missions', required=True,
                        help='Name of the misions file to be executed.')
    parser.add_argument('--missions-subset', default='', type=str,
                        help='Execute a specific set of missions')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per mission.')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume execution from the last successfully completed mission')
    parser.add_argument('--development', type=bool, default=False,
                        help='Set the Leaderboard to run as development. No results will be computed. Ignores resume argument')
    parser.add_argument('--qualifier', type=bool, default=False,
                        help='Set the Leaderboard to run as the qualifier. This has a smaller map size and mission time')
    parser.add_argument('--evaluation', type=bool, default=False,
                        help='Set the Leaderboard to run as the evaluation. No segmentation cameras are allowed here')
    parser.add_argument('--testing', type=bool, default=False,
                        help='Set the Leaderboard to run in testing mode (useful for validating docker images). This enables evaluation and development modes')

    # agent options
    parser.add_argument("-a", "--agent", type=str, required=True,
                        help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    # output options
    parser.add_argument("--checkpoint", type=str, default='results',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--record', type=bool, default=False,
                        help='Flag to (de)activate the recording of the simulation')
    parser.add_argument('--record-control', type=bool, default=False,
                        help='Flag to (de)activate the recording of the agent control')

    arguments = parser.parse_args()

    if arguments.testing:
        arguments.evaluation = True
        arguments.development = True

    if not arguments.testing and arguments.development and arguments.evaluation:
        raise ValueError("Cannot run the Leaderboard in both development and evaluation mode")

    leaderboard_evaluator = LeaderboardEvaluator()
    crashed = leaderboard_evaluator.run(arguments)

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
