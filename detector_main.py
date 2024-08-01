import argparse
import logging
import torch
import random
import numpy as np
import os
import time
try:
    import wandb
except ImportError:
    pass
from detector.dataset import CustomDataset, preprocess_and_save_images, get_datasets
from detector.deep_svdd import DeepSVDD


if __name__ == "__main__":
    ### Arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action = "store_true", default = False)
    parser.add_argument("--train", action = "store_true", default = False)
    parser.add_argument("--test", action = "store_true", default = False)
    parser.add_argument("--latent", action = "store_true", default = False)  # if not latent then image

    parser.add_argument("--env_name", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = True)
    parser.add_argument("--format", type = str, default = "np", choices = ["h5", "np", "png"])  # only for preprocessing
    parser.add_argument("--exp_name", type = str, default = "DEFAULT")
    parser.add_argument("--objective", type = str, default = "one-class", choices = ["one-class", "soft-boundary"])
    parser.add_argument("--nu", type = float, default = 0.1)
    parser.add_argument("--model_file", type = str, default = None)

    parser.add_argument("--optimizer", type = str, default = "adam")
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--num_epochs", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--weight_decay", type = float, default = 1e-6)

    parser.add_argument("--pretrain", action = "store_true", default = True)
    parser.add_argument("--ae_model_file", type = str, default = None)
    parser.add_argument("--ae_optimizer", type = str, default = "adam")
    parser.add_argument("--ae_lr", type = float, default = 0.001)
    parser.add_argument("--ae_num_epochs", type = int, default = 100)
    parser.add_argument("--ae_batch_size", type = int, default = 128)
    parser.add_argument("--ae_weight_decay", type = float, default = 1e-6)

    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--seed", type = int, default = 8888)
    parser.add_argument("--use_wandb", action = "store_true", default = False)

    args = parser.parse_args()
    assert (args.preprocess and not args.train and not args.test) or (args.train and not args.preprocess and not args.test) or (args.test and not args.preprocess and not args.train), "Must have exactly one of preprocess, train, or test"
    if args.train or args.test:
        assert args.env_name is not None, "Please provide env name"

    ### Seeding ###
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)

    ### Device ###
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.device in ["gpu", "cuda"]:
        device = torch.device("cuda")
        print("Using GPU", args.gpu)
    elif args.device == "cpu":
        device = torch.device("cpu")
        print("Using CPU")
    
    ### Logging setup ###
    if args.preprocess:
        folder_name = "preprocess_detector"
    elif args.train:
        folder_name = "train_detector"
    else:
        folder_name = "test_detector"
    logdir = os.path.join(os.getcwd(), "logs", folder_name, args.env_name, args.exp_name)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S")
    if not args.preprocess:
        run_name += f"__seed_{args.seed}"
    logdir = os.path.join(logdir, run_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if args.use_wandb:
        cfg = vars(args)
        wandb.init(project = "ood-detector", config = cfg, resume = "allow")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(logdir, "log.txt"))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print(f"Logging to {logdir}")

    ### Running things ###
    if args.preprocess:
        logger.info("Starting to preprocess images")
        start = time.time()
        preprocess_and_save_images(args.data_dir, logdir, args.format)
        logger.info(f"Done preprocessing, took {(time.time() - start) / 60} minutes")
    else:
        logger.info("Creating DeepSVDD model(s)...")
        deep_svdd = DeepSVDD(args.objective, args.nu, args.latent)
        deep_svdd.set_network(args.exp_name)
        if args.model_file is not None:
            deep_svdd.load_model(network_save_path = args.model_file)
            logger.info(f"Loaded main network from {args.model_file}")
        logger.info("Done!")

        if args.train:
            train_dataset, valid_dataset = get_datasets(args.data_dir, True)
            if args.pretrain:
                if args.ae_model_file is None:
                    logger.info("Pretraining with the following hyperparameters")
                    logger.info(f"\nOptimizer: {args.ae_optimizer}\nLearning rate: {args.ae_lr}\nNum epochs: {args.ae_num_epochs}\nBatch size: {args.ae_batch_size}\nWeight decay: {args.ae_weight_decay}")
                    deep_svdd.pretrain(train_dataset, valid_dataset, optimizer_name = args.ae_optimizer, lr = args.ae_lr, num_epochs = args.ae_num_epochs, batch_size = args.ae_batch_size, weight_decay = args.ae_weight_decay, device = device)
                    deep_svdd.save_model(ae_save_path = os.path.join(logdir, "autoencoder.tar"))
                    logger.info(f"Saved autoencoder tar to {os.path.join(logdir, 'autoencoder.tar')}")
                else:
                    deep_svdd.load_model(ae_save_path = args.ae_model_file)
                    logger.info(f"Loaded autoencoder from {args.ae_model_file}")
        
            deep_svdd.init_network_weights()  # copy encoder to network
            deep_svdd.train(train_dataset, valid_dataset, optimizer_name = args.optimizer, lr = args.lr, num_epochs = args.num_epochs, batch_size = args.batch_size, weight_decay = args.weight_decay, device = device)
            deep_svdd.save_model(network_save_path = os.path.join(logdir, "network.tar"))
            logger.info(f"Saved main network tar to {os.path.join(logdir, 'network.tar')}")
        elif args.test:
            test_dataset, _ = get_datasets(args.data_dir, False)
            deep_svdd.test(test_dataset, device = device)
            deep_svdd.save_results(os.path.join(logdir, "results.json"))
            logger.info(f"Saved results to {os.path.join(logdir, 'results.json')}")
