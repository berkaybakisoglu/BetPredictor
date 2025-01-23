"""Main script for running the betting prediction system."""
import argparse
from pathlib import Path
import logging
from datetime import datetime

from src.config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.predictor import UnifiedPredictor
from src.models.ensemble_predictor import EnsemblePredictor
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
        help='Run in test mode with reduced dataset'
    )
    
    parser.add_argument(
        '--predictor',
        type=str,
        choices=['unified', 'ensemble', 'both'],
        default='both',
        help='Which predictor to use (default: both)'
    )
    
    return parser.parse_args()

def run_predictor(predictor_name: str, predictor, config: Config, train_data: pd.DataFrame, 
                 test_data: pd.DataFrame, args) -> None:
    """Run a single predictor through training and evaluation."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*20} Running {predictor_name} Predictor {'='*20}")
    
    if args.train:
        # Train models
        logger.info("Training models...")
        metrics = predictor.train(train_data, save_path=config.models_dir / predictor_name, 
                                test_mode=args.test_mode)
        logger.info(f"{predictor_name} Training metrics: %s", metrics)
    else:
        # Load existing models
        logger.info("Loading existing models...")
        predictor.load_models(config.models_dir / predictor_name)
    
    if args.evaluate:
        # Make predictions on test set
        logger.info("Making predictions...")
        predictions = predictor.predict(test_data)
        
        # Evaluate performance
        logger.info("Evaluating performance...")
        evaluator = BettingEvaluator(config.betting)
        metrics = evaluator.evaluate_predictions(predictions, test_data)
        logger.info(f"{predictor_name} Evaluation metrics: %s", metrics)
        
        # Plot results
        logger.info("Plotting results...")
        evaluator.plot_performance(save_path=config.output_dir / f"performance_{predictor_name}")

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
        
        # Load and preprocess data
        logger.info("Loading data...")
        train_data, test_data = data_loader.load_data()
        
        if args.test_mode:
            logger.info("Running in test mode with reduced dataset...")
            seasons = sorted(train_data['Season'].unique())
            if len(seasons) > 2:
                test_seasons = seasons[-2:]
                logger.info(f"Using only seasons: {test_seasons}")
                train_data = train_data[train_data['Season'].isin(test_seasons)]
                test_data = test_data[test_data['Season'] >= test_seasons[0]]
        
        # Create features
        logger.info("Engineering features...")
        train_data = feature_engineer.create_features(train_data, is_training=True)
        test_data = feature_engineer.create_features(test_data, is_training=False)
        
        # Run selected predictor(s)
        if args.predictor in ['unified', 'both']:
            unified_predictor = UnifiedPredictor(config.model)
            run_predictor('unified', unified_predictor, config, train_data, test_data, args)
        
        if args.predictor in ['ensemble', 'both']:
            ensemble_predictor = EnsemblePredictor(config.model)
            run_predictor('ensemble', ensemble_predictor, config, train_data, test_data, args)
    
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    main() 