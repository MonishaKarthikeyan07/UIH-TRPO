import os
import torch
from trpo import TRPOAgent
import shutil
import sys

class Trainer:
    def __init__(self):
        self.trpo_agent = TRPOAgent()

    def main(self, ori_dirs, ucc_dirs):
        batch_size = 32
        n_workers = 2
        epochs = 50

        dataloader = self.trpo_agent.collect_samples(ori_dirs, ucc_dirs, batch_size, n_workers)

        for epoch in range(epochs):
            for batch in dataloader:
                states, actions, rewards = batch
                loss = self.trpo_agent.train(ori_dirs, ucc_dirs, batch_size, n_workers, epochs)

            # Save checkpoint
            is_best = False  # Modify this according to your criteria
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.trpo_agent.policy.state_dict(),
                'optimizer': self.trpo_agent.optimizer.state_dict(),
            }, is_best)

        print("Training complete.")

def save_checkpoint(state, is_best):
    """Saves checkpoint to disk"""
    freq = 500
    epoch = state['epoch'] 

    filename = './checkpoints/model_tmp.pth.tar'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    torch.save(state, filename)

    if epoch % freq == 0:
        shutil.copyfile(filename, './checkpoints/model_{}.pth.tar'.format(epoch))
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best_{}.pth.tar'.format(epoch))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER")
        sys.exit(1)

    # Extract command-line arguments
    _, ori_dirs, ucc_dirs = sys.argv

    trainer = Trainer()
    trainer.main(ori_dirs, ucc_dirs)
