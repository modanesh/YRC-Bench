import argparse
import torch
import time
try:
    import wandb
except ImportError:
    pass
from detector.dataset import CustomDataset, preprocess_and_save_images
from detector.deep_svdd import DeepSVDD


if __name__ == "__main__":
    ### Arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action = "store_true", default = False)
    parser.add_argument("--train", action = "store_true", default = False)
    parser.add_argument("--test", action = "store_true", default = False)

    parser.add_argument("--env_name", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = True)
    parser.add_argument("--save_dir", type = str)  # only for preprocessing
    parser.add_argument("--exp_name", type = str, default = "training")
    parser.add_argument("--objective", type = str, default = "one-class", choices = ["one-class", "soft-boundary"])
    parser.add_argument("--nu", type = float, default = 0.1)
    parser.add_argument("--model_file", type = str, default = None)

    parser.add_argument("--optimizer", type = str, default = "adam")
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--num_epochs", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--weight_decay", type = float, default = 1e-6)

    parser.add_argument("--pretrain", action = "store_true", default = True)
    parser.add_argument("--ae_optimizer", type = str, default = "adam")
    parser.add_argument("--ae_lr", type = float, default = 0.001)
    parser.add_argument("--ae_num_epochs", type = int, default = 100)
    parser.add_argument("--ae_batch_size", type = int, default = 128)
    parser.add_argument("--ae_weight_decay", type = float, default = 1e-6)

    parser.add_argument("--device", type = str, default = "cpu")
    parser.add_argument("--gpu", type = int)
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
    logdir = os.path.join(os.getcwd(), folder_name, args.env_name)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f"__seed_{args.seed}"
    logdir = os.path.join(logdir, run_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if args.use_wandb:
        cfg = vars(args)
        wandb.init(project = "ood-detector", config = cfg, resume = "allow")
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(logdir, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(file_handler)
    print(f"Logging to {logdir}")

    ### Running things ###
    if args.preprocess:
        logger.info("Starting to preprocess images")
        preprocess_and_save_images(args.data_dir, args.save_dir)
        logger.info("Done preprocessing")
    elif args.train:
        logger.info("Creating DeepSVDD model(s)...")
        deep_svdd = DeepSVDD(args.objective, args.nu)
        deep_svdd.set_network(args.exp_name)
        if args.model_file is not None:
            deep_svdd.load_model(args.model_file, load_ae = True)
            logger.info(f"Loaded model from {args.model_file}")
        logger.info("Done!")

        train_dataset = CustomDataset(args.data_dir)

        if args.pretrain:
            logger.info("Pretraining with the following hyperparameters")
            logger.info(f"Optimizer: {args.ae_optimizer}\nLearning rate: {args.ae_lr}\nNum epochs: {args.ae_num_epochs}\nBatch size: {args.ae_batch_size}\nWeight decay: {args.ae_weight_decay}")
            deep_svdd.pretrain(train_dataset, optimizer_name = args.ae_optimizer, lr = args.ae_lr, num_epochs = args.ae_num_epochs, batch_size = args.ae_batch_size, weight_decay = args.ae_weight_decay, device = device)
        deep_svdd.train(train_dataset, optimizer_name = args.optimizer, lr = args.lr, num_epochs = args.num_epochs, batch_size = args.batch_size, weight_decay = args.weight_decay, device = device)
        deep_svdd.save_results(os.path.join(logdir, "results.json"))
        deep_svdd.save_model(os.path.join(logdir, "model.tar"))
