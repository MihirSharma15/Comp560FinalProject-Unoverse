"""Convenience script to run agent comparison from project root."""

from src.comparison.compare_agents import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    seed = None if args.seed == -1 else args.seed
    
    main(
        n_train_episodes=args.train_episodes,
        n_eval_episodes=args.eval_episodes,
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        force_retrain=args.force_retrain,
        seed=seed,
        verbose=not args.quiet
    )

