import torch
import torch.optim as optim
import numpy as np
import logging
import time
from sklearn.metrics import roc_auc_score
from detector.dataset import get_dataloader
import wandb


def dynamic_permute(inputs):
    if inputs.dim() < 3:  # latent rep
        return inputs.squeeze()
    batch_size, dim1, dim2, dim3 = inputs.shape
    if dim1 == 3:
        return inputs
    elif dim2 == 3:
        return inputs.permute(0, 2, 1, 3)
    elif dim3 == 3:
        return inputs.permute(0, 3, 1, 2)


class BaseTrainer:
    def __init__(
            self,
            optimizer_name,
            lr,
            num_epochs,
            lr_milestones,
            batch_size,
            weight_decay,
            device,
            num_jobs_dataloader,
            use_wandb = False
    ):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.num_epochs = num_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.num_jobs_dataloader = num_jobs_dataloader
        self.use_wandb = use_wandb
    

class DeepSVDDTrainer(BaseTrainer):
    def __init__(
            self,
            objective,
            radius,
            center,
            nu,
            optimizer_name = "adam",
            lr = 0.001,
            num_epochs = 150,
            lr_milestones = (),
            batch_size = 128,
            weight_decay = 1e-6,
            device = "cuda",
            num_jobs_dataloader = 0,
            use_wandb = False
    ):
        super().__init__(optimizer_name, lr, num_epochs, lr_milestones, batch_size, weight_decay, device, num_jobs_dataloader, use_wandb = use_wandb)
        self.objective = objective
        self.radius = torch.tensor(radius, device = self.device)
        self.center = torch.tensor(center, device = self.device) if center is not None else None
        self.nu = nu
        self.num_warmup_epochs = 10  # for soft-boundary
        self.train_time = None
        self.test_time = None
        self.test_auc = None
        self.test_scores = None

    def train(self, train_dataset, valid_dataset, model):
        logger = logging.getLogger()
        model = model.to(self.device)
        train_loader = get_dataloader(train_dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        if valid_dataset is not None:
            valid_loader = get_dataloader(valid_dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay, amsgrad = self.optimizer_name == "amsgrad")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.lr_milestones, gamma = 0.1)
        
        if self.center is None:
            logger.info("Center was None, initializing now...")
            self.center = self.init_center(train_loader, model)
            logger.info("Initialized center")
        
        logger.info("Starting to train Deep-SVDD...")
        start_time = time.time()
        best_loss = np.inf
        patience = 10
        patience_counter = 0
        for epoch in range(self.num_epochs):
            model.train()
            if epoch in self.lr_milestones:
                logger.info("  LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))
                if self.use_wandb:
                    wandb.log({"epoch": epoch, "lr": float(scheduler.get_lr()[0])})
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = dynamic_permute(inputs.float().to(self.device))
                optimizer.zero_grad()
                outputs = model(inputs)
                dist = torch.sum((outputs - self.center)**2, dim = 1)
                if self.objective == "soft-boundary":
                    scores = dist - self.radius**2
                    loss = self.radius**2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                if self.objective == "soft-boundary" and epoch >= self.num_warmup_epochs:
                    self.radius.data = torch.tensor(self.get_radius(dist, self.nu), device = self.device)
                epoch_loss += loss.item()
                num_batches += 1
            epoch_train_time = time.time() - epoch_start
            scheduler.step()
            logger.info("  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}" .format(epoch + 1, self.num_epochs, epoch_train_time, epoch_loss / num_batches))
            if self.use_wandb:
                wandb.log({"epoch": epoch + 1, "train_time": epoch_train_time, "avg_loss": epoch_loss / num_batches})
            
            if valid_dataset is not None:
                model.eval()
                val_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for data in valid_loader:
                        inputs, _, _ = data
                        inputs = dynamic_permute(inputs.float().to(self.device))
                        outputs = model(inputs)
                        dist = torch.sum((outputs - self.center)**2, dim = 1)
                        if self.objective == "soft-boundary":
                            scores = dist - self.radius**2
                            loss = self.radius**2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                        else:
                            loss = torch.mean(dist)
                        val_loss += loss.item()
                        num_val_batches += 1
                avg_val_loss = val_loss / num_val_batches
                logger.info("  Validation Loss: {:.8f}".format(avg_val_loss))
                if self.use_wandb:
                    wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered after epoch {}".format(epoch + 1))
                    break

        self.train_time = time.time() - start_time
        logger.info("Training time: %.3f" % self.train_time)
        logger.info("Finished training Deep-SVDD!!!!!")
        if valid_dataset is not None:
            model.load_state_dict(best_model_state)
        return model
    
    def test(self, dataset, model):
        logger = logging.getLogger()
        model = model.to(self.device)
        test_loader = get_dataloader(dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        
        logger.info("Starting inference...")
        start_time = time.time()
        idx_label_score = []
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = dynamic_permute(inputs.float().to(self.device))
                outputs = model(inputs)
                dist = torch.sum((outputs - self.center)**2, dim = 1)
                if self.objective == "soft-boundary":
                    scores = dist - self.radius**2
                else:
                    scores = dist
                idx_label_score.extend(list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist())))
        self.test_time = time.time() - start_time
        logger.info("Testing time: %.3f" % self.test_time)
        
        _, labels, scores = zip(*idx_label_score)
        self.test_scores = scores
        try:
            self.test_auc = roc_auc_score(np.array(labels), np.array(scores))
        except ValueError:  # only one class => cannot calculate AUC
            self.test_auc = -1
        logger.info("Test set AUC: {:.2f}%".format(100. * self.test_auc))
        logger.info("Finished inference!!!!!")
    
    def init_center(self, train_loader, model, eps = 0.1):
        num_samples = 0
        center = torch.zeros(model.last_layer_dim, device = self.device)
        model.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _ = data
                inputs = dynamic_permute(inputs.float().to(self.device))
                outputs = model(inputs)
                num_samples += outputs.shape[0]
                center += torch.sum(outputs, dim = 0)
        center /= num_samples
        # If c_i is too close to 0, set to ±eps, because a zero unit can be trivially matched with zero weights
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        return center
    
    def get_radius(self, dist, nu):
        # Use (1 - nu) quantile of distances to solve for radius
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


class AETrainer(BaseTrainer):
    def __init__(
            self,
            optimizer_name = "adam",
            lr = 0.001,
            num_epochs = 150,
            lr_milestones = (),
            batch_size = 128,
            weight_decay = 1e-6,
            device = "cuda",
            num_jobs_dataloader = 0,
            use_wandb = False
    ):
        super().__init__(optimizer_name, lr, num_epochs, lr_milestones, batch_size, weight_decay, device, num_jobs_dataloader, use_wandb = use_wandb)
    
    def train(self, train_dataset, valid_dataset, model):
        logger = logging.getLogger()
        model = model.to(self.device)
        train_loader = get_dataloader(train_dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        if valid_dataset is not None:
            valid_loader = get_dataloader(valid_dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay, amsgrad = self.optimizer_name == "amsgrad")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.lr_milestones, gamma = 0.1)

        logger.info("Starting to train autoencoder...")
        start_time = time.time()
        best_loss = np.inf
        patience = 10
        patience_counter = 0
        for epoch in range(self.num_epochs):
            model.train()
            if epoch in self.lr_milestones:
                logger.info("  LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))
                if self.use_wandb:
                    wandb.log({"epoch": epoch, "lr": float(scheduler.get_lr()[0])})
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = dynamic_permute(inputs.float().to(self.device))
                optimizer.zero_grad()
                outputs = model(inputs)
                scores = torch.sum((outputs - inputs)**2, dim = tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            epoch_train_time = time.time() - epoch_start
            scheduler.step()
            logger.info("  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}" .format(epoch + 1, self.num_epochs, epoch_train_time, epoch_loss / num_batches))
            if self.use_wandb:
                wandb.log({"epoch": epoch + 1, "train_time": epoch_train_time, "avg_loss": epoch_loss / num_batches})
            
            if valid_dataset is not None:
                model.eval()
                val_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for data in valid_loader:
                        inputs, _, _ = data
                        inputs = dynamic_permute(inputs.float().to(self.device))
                        outputs = model(inputs)
                        scores = torch.sum((outputs - inputs)**2, dim = tuple(range(1, outputs.dim())))
                        loss = torch.mean(scores)
                        val_loss += loss.item()
                        num_val_batches += 1
                avg_val_loss = val_loss / num_val_batches
                logger.info("  Validation Loss: {:.8f}" .format(avg_val_loss))
                if self.use_wandb:
                    wandb.log({"epoch": epoch + 1, "avg_val_loss": avg_val_loss})
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered after epoch {}".format(epoch + 1))
                    break

        self.train_time = time.time() - start_time
        logger.info("Pretraining time: %.3f" % self.train_time)
        logger.info("Finished pretraining!!!!!")
        if valid_dataset is not None:
            model.load_state_dict(best_model_state)
        return model
    
    def test(self, dataset, model):
        logger = logging.getLogger()
        model = model.to(self.device)
        test_loader = get_dataloader(dataset, batch_size = self.batch_size, num_workers = self.num_jobs_dataloader)
        
        logger.info("Starting autoencoder inference...")
        test_loss = 0.0
        num_batches = 0
        start_time = time.time()
        idx_label_score = []
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                scores = torch.sum((outputs - inputs)**2, dim = tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                idx_label_score.extend(list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist())))
                test_loss += loss.item()
                num_batches += 1
        logger.info("Inference loss: {:.8f}".format(test_loss / num_batches))
        self.test_time = time.time() - start_time
        logger.info("Autoencoder testing time: %.3f" % self.test_time)

        _, labels, scores = zip(*idx_label_score)
        auc = roc_auc_score(np.array(labels), np.array(scores))
        logger.info("Test set AUC: {:.2f}%".format(100. * auc))
        logger.info("Finished autoencoder inference!!!!!")
