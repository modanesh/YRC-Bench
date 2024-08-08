import json
import torch
from detector.models import Network, AutoEncoder
from detector.trainers import DeepSVDDTrainer, AETrainer
import numpy as np


class DeepSVDD:
    """
    Class for the Deep-SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        radius: Hypersphere radius R.
        center: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective = "one-class", nu = 0.1, for_latent = False):
        self.objective = objective
        self.nu = nu
        self.for_latent = for_latent
        self.radius = 0.0
        self.center = None
        self.net_name = None
        self.net = None
        self.trainer = None
        self.optimizer_name = None
        self.ae_net = None
        self.AETrainer = None
        self.ae_optimizer_name = None
        self.results = {
            "train_time": None,
            "test_time": None,
            "test_auc": None,
            "test_scores": None,
            "test_thresholds": None
        }
    
    def set_network(self, net_name):
        """
        Builds neural network
        """
        self.net_name = net_name
        self.net = Network(self.for_latent)
    
    def train(self, train_dataset, valid_dataset, optimizer_name = "adam", lr = 0.001, num_epochs = 50, lr_milestones = (), batch_size = 128, weight_decay = 1e-6, device = "cuda", num_jobs_dataloader = 0):
        """
        Trains Deep-SVDD model
        """
        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.radius, self.center, self.nu, self.optimizer_name, lr = lr, num_epochs = num_epochs, lr_milestones = lr_milestones, batch_size = batch_size, weight_decay = weight_decay, device = device, num_jobs_dataloader = num_jobs_dataloader)
        self.net = self.trainer.train(train_dataset, valid_dataset, self.net)
        self.radius = float(self.trainer.radius.cpu().data.numpy())
        self.center = self.trainer.center.cpu().data.numpy().tolist()
        self.results["train_time"] = self.trainer.train_time
    
    def test(self, dataset, device = "cuda", num_jobs_dataloader = 0):
        """
        Runs inference on Deep-SVDD model
        """
        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.radius, self.center, self.nu, device = device, num_jobs_dataloader = num_jobs_dataloader)
        self.trainer.test(dataset, self.net)
        self.results["test_time"] = self.trainer.test_time
        self.results["test_auc"] = self.trainer.test_auc
        self.results["test_scores"] = self.trainer.test_scores
        ood_boundary = np.max(self.trainer.test_scores)  # max distance from center for training datapoint
        percentiles = [np.percentile(self.trainer.test_scores, p) for p in range(50, 91, 10)]
        percentile_diffs = [abs(ood_boundary - perc) for perc in percentiles][::-1]
        pseudo_percentiles = range(50, 151, 10)
        thresholds = []
        perc_diff_idx = 0
        for i, pp in enumerate(pseudo_percentiles):
            if pp < 100:
                thresholds.append(percentiles[i])
            elif pp == 100:
                thresholds.append(ood_boundary)
            else:
                thresholds.append(ood_boundary + percentile_diffs[0])
                perc_diff_idx += 1
        self.results["test_thresholds"] = {k: v for k, v in zip(pseudo_percentiles, thresholds)}
        addl_percentiles = [1, 5] + list(range(10, 41, 10))
        self.results["test_thresholds"].update({k: v for k, v in zip(addl_percentiles, [np.percentile(self.trainer.test_scores, p) for p in addl_percentiles])})
    
    def pretrain(self, train_dataset, valid_dataset, optimizer_name = "adam", lr = 0.001, num_epochs = 100, lr_milestones = (), batch_size = 128, weight_decay = 1e-6, device = "cuda", num_jobs_dataloader = 0):
        """
        Pretrains the weights for the Deep-SVDD model via an autoencoder
        """
        if self.ae_net is None:
            self.ae_net = AutoEncoder(self.for_latent)
        self.ae_optimizer_name = optimizer_name
        self.AETrainer = AETrainer(optimizer_name, lr = lr, num_epochs = num_epochs, lr_milestones = lr_milestones, batch_size = batch_size, weight_decay = weight_decay, device = device, num_jobs_dataloader = num_jobs_dataloader)
        self.ae_net = self.AETrainer.train(train_dataset, valid_dataset, self.ae_net)
        # self.AETrainer.test(dataset, self.ae_net)
    
    def init_network_weights(self):
        """
        Initializes Deep-SVDD model weights from pretrained autoencoder's encoder weights
        """
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}  # only get encoder weights
        net_dict.update(ae_net_dict)  # overwrite/update weights
        self.net.load_state_dict(net_dict)
    
    def save_model(self, network_save_path = None, ae_save_path = None):
        """
        Saves actual network and autoencoder
        """
        if network_save_path:
            torch.save({
                "radius": self.radius,
                "center": self.center,
                "net_dict": self.net.state_dict()
            }, network_save_path)
        if ae_save_path:
            torch.save({
                "ae_net_dict": self.ae_net.state_dict()
            }, ae_save_path)
    
    def load_model(self, network_save_path = None, ae_save_path = None):
        """
        Loads Deep-SVDD model
        """
        if network_save_path:
            model_info = torch.load(network_save_path)
            self.radius = model_info["radius"]
            self.center = model_info["center"]
            self.net.load_state_dict(model_info["net_dict"])
        if ae_save_path:
            model_info = torch.load(ae_save_path)
            if self.ae_net is None:
                self.ae_net = AutoEncoder(self.for_latent)
            self.ae_net.load_state_dict(model_info["ae_net_dict"])
    
    def save_results(self, out_file):
        """
        Saves results to JSON file
        """
        with open(out_file, "w") as f:
            json.dump(self.results, f)

