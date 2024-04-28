import argparse
from trpo import TRPOAgent

def main(ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
    trpo_agent = TRPOAgent()
    trpo_agent.train(ori_dirs, ucc_dirs, batch_size, n_workers, epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRPO model for underwater image enhancement.')
    parser.add_argument('ori_dirs', metavar='ori_dirs', type=str, nargs='+', help='path to original image directories')
    parser.add_argument('ucc_dirs', metavar='ucc_dirs', type=str, nargs='+', help='path to underwater corrected image directories')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--n_workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    args = parser.parse_args()

    main(args.ori_dirs, args.ucc_dirs, args.batch_size, args.n_workers, args.epochs)
