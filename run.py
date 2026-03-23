import argparse

from src.common import get_logger, load_config
from src.pipeline import run_inference, run_summary, run_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True, choices=['update', 'inference', 'summary'])
    parser.add_argument('-file', default=None)
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger()

    if args.mode == 'update':
        print(run_update(cfg, logger))
    elif args.mode == 'inference':
        if args.file is None:
            raise ValueError('Для режима inference нужен путь к файлу через -file')
        print(run_inference(cfg, logger, args.file))
    elif args.mode == 'summary':
        print(run_summary(cfg, logger))


if __name__ == '__main__':
    main()
