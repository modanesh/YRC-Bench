import jsonargparse


def make():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to YAML config file")
    parser.add_argument("--name", type=str, help="name of this run")
    parser.add_argument(
        "--coord_policy.agent",
        type=str,
        choices=["weak", "strong"],
        help="always choose action of this agent",
    )

    args = parser.parse_args()

    return args
