import torch
from uwcc import UWCCDataset
from model import PhysicalNN
from torch.utils.data import DataLoader

class TRPOAgent:
    def __init__(self):
        self.policy = PhysicalNN()  # Define your policy network
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    def collect_samples(self, ori_dirs, ucc_dirs, batch_size, n_workers):
        train_set = UWCCDataset(ori_dirs, ucc_dirs, train=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        return train_loader

    def surrogate_loss(self, old_probs, new_probs, advantages):
        eps = torch.finfo(old_probs.dtype).eps
        old_probs = torch.clamp(old_probs, min=eps)

        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages

        return -torch.min(surr1, surr2).mean()

    def compute_advantages(self, rewards):
        rewards = rewards.float()
        return rewards - rewards.mean()

    def train(self, ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
        torch.autograd.set_detect_anomaly(True)
        dataloader = self.collect_samples(ori_dirs, ucc_dirs, batch_size, n_workers)

        for epoch in range(epochs):
            for batch in dataloader:
                states, actions, rewards = batch
                old_probs = self.policy(states)
                rewards = rewards.float()
                advantages = self.compute_advantages(rewards)

                for _ in range(10):
                    new_probs = self.policy(states)
                    loss = self.surrogate_loss(old_probs, new_probs, advantages)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

        return loss.item()
