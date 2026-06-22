import argparse
import sys
import os
import pyaudio
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    logger.remove()
    os.makedirs("logs", exist_ok=True)
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add("logs/session_{time:YYYY-MM-DD_HH-mm-ss}.log", level="DEBUG", rotation="50 MB", retention="7 days")


def list_devices():
    p = pyaudio.PyAudio()
    print("\nNo input device specified. Printing list of input devices now:")
    for i in range(p.get_device_count()):
        print(f"Device number ({i}): {p.get_device_info_by_index(i).get('name')}")
    print("\nRun this program with -input 1, or the number of the input you'd like to use.\n")
    p.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="Archer-Robot Vocal Collaboration System")
    parser.add_argument("-input", required=False, type=int, help="Audio Input Device")
    parser.add_argument("-interval", default=None, choices=["unison", "third", "fifth", "octave"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    if not args.input and args.input != 0:
        list_devices()
        sys.exit(0)

    from config.config_loader import get_config
    cfg = get_config()
    cfg["audio"]["input_device"] = args.input

    logger.info("=" * 60)
    logger.info("  Archer-Robot Vocal Collaboration System")
    logger.info("=" * 60)
    logger.info(f"Using input device: {args.input}")

    from core.pipeline import Pipeline
    pipeline = Pipeline()

    if args.interval:
        pipeline.set_interval(args.interval)

    logger.info("Pipeline running — play music or sing now!")
    pipeline.start()


if __name__ == "__main__":
    main()