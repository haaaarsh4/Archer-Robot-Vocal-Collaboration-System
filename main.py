import argparse
import sys
import os
from loguru import logger

# Adding this current folder to Python's search path
sys.path.insert(0, os.path.dirname(__file__))

# Logger to show info / debug message's on the command line when the program is run and also store debug log's in /logs/
def setup_logging(debug):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add("logs/session_{time:YYYY-MM-DD_HH-mm-ss}.log",
               level="DEBUG", rotation="50 MB", retention="7 days")

    os.makedirs("logs", exist_ok=True)

# Defining what command line flags the user can pass to the program
def parse_args():
    parser = argparse.ArgumentParser(
        description="Archer-Robot Vocal Collaboration System"
    )

    # prints the list of audio devices connected to the device
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit"
    )
    
    # Harmony iterval to cutomize how far of a note we want the note to be from what Archer is singing
    parser.add_argument(
        "--interval", default=None,
        choices=["unison", "third", "fifth", "octave"],
        help="Harmony interval (default: from config)"
    )
    
    # turns on verbose logging to make debuging easier
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(debug=args.debug)

    logger.info("=" * 60)
    logger.info("  Archer-Robot Vocal Collaboration System")
    logger.info("=" * 60)

    if args.list_devices:
        from core.audio_capture import AudioCapture
        AudioCapture().list_devices()
        sys.exit(0)

    from config.config_loader import get_config, reload_config
    cfg = get_config()
    logger.info(f"Synthesis engine : {cfg['synthesis']['engine']}")
    logger.info(f"Pitch engine     : {cfg['pitch']['engine']}")
    logger.info(f"Cree tokenizer   : {'enabled' if cfg['cree_tokenizer']['enabled'] else 'disabled'}")
    logger.info("-" * 60)

    from core.pipeline import Pipeline
    pipeline = Pipeline()

    if args.interval:
        pipeline.set_interval(args.interval)
        logger.info(f"Harmony interval set to: {args.interval}")

    logger.info("Starting pipeline!")
    pipeline.start()


if __name__ == "__main__":
    main()