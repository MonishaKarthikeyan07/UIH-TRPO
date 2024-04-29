import argparse
from trpo import TRPOAgent

def main():
    parser = argparse.ArgumentParser(description='Training TRPO Agent')
    parser.add_argument('--ori_dirs', type=str, nargs='+', help='Directories containing original images')
    parser.add_argument('--ucc_dirs', type=str, nargs='+', help='Directories containing UCC images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    trpo_agent = TRPOAgent()
    trpo_agent.train(args.ori_dirs, args.ucc_dirs, args.batch_size, args.n_workers, args.epochs)

if __name__ == '__main__':
    main()
