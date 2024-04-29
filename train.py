import argparse
import torch
from trpo import TRPOAgent
from uwcc import uwcc  # Import the uwcc class from uwcc.py

def main(ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
    trpo_agent = TRPOAgent()
    try:
        trpo_agent.train(ori_dirs, ucc_dirs, batch_size, n_workers, epochs)
    except RuntimeError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRPO model for underwater image enhancement.')
    parser.add_argument('ori_dirs', metavar='ori_dirs', type=str, nargs='+', help='path to original image directories')
    parser.add_argument('ucc_dirs', metavar='ucc_dirs', type=str, nargs='+', help='path to underwater corrected image directories')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--n_workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    args = parser.parse_args()

    print("Original image directories:", args.ori_dirs)
    print("Underwater corrected image directories:", args.ucc_dirs)

    main(args.ori_dirs, args.ucc_dirs, args.batch_size, args.n_workers, args.epochs)
