import argparse, json

from watchdog.watchdog import MultiCamWatchdog

parser = argparse.ArgumentParser(description="Run watchdog: Initialize detector, cameras and communication bot, according to the configuration. "
                                             " Create empty file 'stop' in the output dir to interrupt the process. See ReadMe.md for help!")
parser.add_argument("config_json_path", type=str, help="Path to the main config file")
args = parser.parse_args()

with open(args.config_json_path, "r") as file:
    config_dict = json.load(file)

# Set and run watchdog!
MultiCamWatchdog(config_dict)
