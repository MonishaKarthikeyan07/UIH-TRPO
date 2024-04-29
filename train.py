import argparse
from trpo import TRPOAgent

def main():
    parser = argparse.ArgumentParser(description="TRPO training script")
    parser.add_argument("ori_dirs", nargs='+', help="List of directories containing original images")
    parser.add_argument("ucc_dirs", nargs='+', help="List of directories containing UCC images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    trpo_agent = TRPOAgent()
    trpo_agent.train(args.ori_dirs, args.ucc_dirs, args.batch_size, args.n_workers, args.epochs)

if __name__ == "__main__":
    main()
