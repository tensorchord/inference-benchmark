from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--service", "-s", required=True, choices=("mosec"))

    return parser


def run():
    parser = build_parser()
    args = parser.parse_args()
    print(args.service)
