




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Main entry point for training and evaluating the model")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model-version', type=str, default='original', help='Model version to evaluate')
    args = parser.parse_args()

    if args.train:
        from train_xgboost import main as train_main
        train_main()
    elif args.evaluate:
        from evaluate_model import evaluate_model
        evaluate_model(model_version=args.model_version)
    else:
        print("Please specify --train or --evaluate")
if __name__ == "__main__":
    main()