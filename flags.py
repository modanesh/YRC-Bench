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

    # ood policy
    parser.add_argument(
        "--coord_policy.method",
        type=str,
        choices=["DeepSVDD"],
        help="method for detecting OOD samples"
    )

    parser.add_argument('--env_name', required=True, type=str, help='name of the environment')
    parser.add_argument('--agent_sim_weak', required=True, type=str, help='path to the sim weak agent')
    parser.add_argument('--agent_weak', required=True, type=str, help='path to the weak agent')
    parser.add_argument('--agent_strong', required=False, type=str, help='path to the strong agent')

    parser.add_argument('--file_name', required=False, type=str, help='file name for evaluation')

    args = parser.parse_args()

    return args
