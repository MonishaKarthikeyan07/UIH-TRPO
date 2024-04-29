import argparse
from trpo import TRPOAgent

def main():
    parser = argparse.ArgumentParser(description='Train TRPO Agent')
    parser.add_argument('ori_dirs', type=str, help='Path to original image directory')
    parser.add_argument('ucc_dirs', type=str, help='Path to transformed image directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    args = parser.parse_args()

    trpo_agent = TRPOAgent()
    trpo_agent.train(args.ori_dirs, args.ucc_dirs, args.batch_size, args.n_workers, args.epochs)

if __name__ == '__main__':
    main()
