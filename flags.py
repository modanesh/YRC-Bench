import jsonargparse


def make():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to YAML config file")
    parser.add_argument("--name", type=str, help="name of this run")
    parser.add_argument("--overwrite", action="store_true", help="overwrite experiment folder (if exists")

    # always policy
    parser.add_argument(
        "--coord_policy.agent",
        type=str,
        choices=["weak", "strong"],
        help="always choose action of this agent",
    )

    # threshold policy
    parser.add_argument(
        "--coord_policy.metric",
        type=str,
        choices=["max_logit", "max_prob", "margin", "neg_entropy", "neg_energy"],
        help="metric for computing scores"
    )


    args = parser.parse_args()

    return args
