import jsonargparse
import torch


def make():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("-config", type=str, help="path to config file")

    args = parser.parse_args()

    return args
