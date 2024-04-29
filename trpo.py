import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from uwcc import uwcc  # Import uwcc class from uwcc.py
from model import PhysicalNN  # Import PhysicalNN from model.py

class TRPOAgent:
    def __init__(self):
        self.policy = PhysicalNN()  # Define your policy network
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    def collect_samples(self, ori_dirs, ucc_dirs, batch_size, n_workers):
        train_set = uwcc(ori_dirs, ucc_dirs, train=True)  # Use uwcc class instead of UWCCDataset
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        return train_loader

    def surrogate_loss(self, old_probs, new_probs, advantages):
        # Check for NaN or infinite values in input tensors
        if torch.isnan(old_probs).any() or torch.isinf(old_probs).any():
            raise ValueError("old_probs tensor contains NaN or infinite values.")
        if torch.isnan(new_probs).any() or torch.isinf(new_probs).any():
            raise ValueError("new_probs tensor contains NaN or infinite values.")
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            raise ValueError("advantages tensor contains NaN or infinite values.")

        # Add a small epsilon value to avoid division by zero
        eps = torch.finfo(old_probs.dtype).eps
        old_probs = torch.clamp(old_probs, min=eps)

        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages

        # Check for NaN or infinite values in intermediate tensors
        if torch.isnan(surr1).any() or torch.isinf(surr1).any():
            raise ValueError("surr1 tensor contains NaN or infinite values.")
        if torch.isnan(surr2).any() or torch.isinf(surr2).any():
            raise ValueError("surr2 tensor contains NaN or infinite values.")

        return -torch.min(surr1, surr2).mean()

    def compute_advantages(self, rewards):
        rewards = rewards.float()  # Convert to floating-point dtype
        return rewards - rewards.mean()

    def train(self, ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
        dataloader = self.collect_samples(ori_dirs, ucc_dirs, batch_size, n_workers)

        for epoch in range(epochs):
            for batch in dataloader:
                states, actions, rewards = batch
                old_probs = self.policy(states)

                # Ensure rewards tensor has appropriate dtype for computing the mean
                rewards = rewards.float()  # Convert to floating-point dtype

                # Compute advantages
                advantages = self.compute_advantages(rewards)

                # Policy gradient ascent
                for _ in range(10):  # TRPO typically uses a line search or conjugate gradient
                    new_probs = self.policy(states)
                    loss = self.surrogate_loss(old_probs, new_probs, advantages)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)  # Add retain_graph=True
                    self.optimizer.step()

        return loss.item()
