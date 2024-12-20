"""Main script for running the betting prediction system."""
import argparse
from pathlib import Path
import logging
from datetime import datetime

from src.config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.predictor import UnifiedPredictor
from src.evaluation.evaluator import BettingEvaluator

def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                output_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run betting prediction system')
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw'),
        help='Directory containing raw data files'
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('models'),
        help='Directory to save/load models'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory to save outputs'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train new models'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model performance'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with reduced dataset (last 2 seasons only)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use in test mode (default: 0.2, i.e., 20%% of the data)'
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function to run the betting prediction system."""
    args = parse_args()
    
    # Initialize config
    config = Config(
        data=DataConfig(data_dir=args.data_dir),
        features=FeatureConfig(),
        model=ModelConfig(),
        betting=BettingConfig(),
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
    
    # Set up logging
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        data_loader = DataLoader(config.data)
        feature_engineer = FeatureEngineer(config.features)
        predictor = UnifiedPredictor(config.model)
        evaluator = BettingEvaluator(config.betting)
        
        # Load and preprocess data
        logger.info("Loading data...")
        train_data, test_data = data_loader.load_data()
        
        if args.test_mode:
            logger.info("Running in test mode with reduced dataset...")
            # Get unique seasons
            seasons = sorted(train_data['Season'].unique())
            
            if len(seasons) > 2:
                # Use only the last 2 seasons for test mode
                test_seasons = seasons[-2:]
                logger.info(f"Using only seasons: {test_seasons}")
                
                # Filter data for test seasons
                train_data = train_data[train_data['Season'].isin(test_seasons)]
                test_data = test_data[test_data['Season'] >= test_seasons[0]]
                
                # Further reduce data size if specified
                if args.test_size < 1.0:
                    train_size = int(len(train_data) * args.test_size)
                    test_size = int(len(test_data) * args.test_size)
                    
                    train_data = train_data.sample(n=train_size, random_state=42)
                    test_data = test_data.sample(n=test_size, random_state=42)
                    
                    logger.info(f"Reduced dataset size - Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Create features
        logger.info("Engineering features...")
        train_data = feature_engineer.create_features(train_data, is_training=True)
        test_data = feature_engineer.create_features(test_data, is_training=False)
        
        if args.train:
            # Train models
            logger.info("Training models...")
            metrics = predictor.train(train_data, save_path=config.models_dir, test_mode=args.test_mode)
            logger.info("Training metrics: %s", metrics)
        else:
            # Load existing models
            logger.info("Loading existing models...")
            predictor.load_models(config.models_dir)
        
        if args.evaluate:
            # Make predictions on test set
            logger.info("Making predictions...")
            predictions = predictor.predict(test_data)
            
            # Evaluate performance
            logger.info("Evaluating performance...")
            metrics = evaluator.evaluate_predictions(predictions, test_data)
            logger.info("Evaluation metrics: %s", metrics)
            
            # Plot results
            logger.info("Plotting results...")
            evaluator.plot_performance(save_path=config.output_dir)
    
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    main() 