import argparse
import os
import sys
import torch
import shutil
from trpo import TRPOAgent
from uwcc import uwcc

class Trainer:
    def __init__(self):
        self.trpo_agent = TRPOAgent()

    def main(self, ori_dirs, ucc_dirs):
        batch_size = 32
        n_workers = 2
        epochs = 50

        try:
            dataloader = self.trpo_agent.collect_samples(ori_dirs, ucc_dirs, batch_size, n_workers)
            for epoch in range(epochs):
                for batch in dataloader:
                    states, actions, rewards = batch
                    loss = self.trpo_agent.train(states, actions, rewards)

                # Save checkpoint
                is_best = False  # Modify this according to your criteria
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trpo_agent.policy.state_dict(),
                    'optimizer': self.trpo_agent.optimizer.state_dict(),
                }, is_best)

            print("Training complete.")
        except RuntimeError as e:
            print(f"Error during training: {str(e)}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('ori_dirs', type=str, help='Directory containing original images')
    parser.add_argument('ucc_dirs', type=str, help='Directory containing UCC images')
    args = parser.parse_args()

    ori_dirs = args.ori_dirs
    ucc_dirs = args.ucc_dirs

    # Check if directories exist
    if not os.path.exists(ori_dirs):
        print(f"Error: Original image directory '{ori_dirs}' not found.")
        sys.exit(1)
    if not os.path.exists(ucc_dirs):
        print(f"Error: UCC image directory '{ucc_dirs}' not found.")
        sys.exit(1)

    trainer = Trainer()
    trainer.main(ori_dirs, ucc_dirs)
