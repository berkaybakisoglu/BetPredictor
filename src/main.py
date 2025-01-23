"""Main script for running the betting prediction system."""
import argparse
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.predictor import UnifiedPredictor
from src.models.hybrid_predictor import HybridPredictor
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
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use in test mode (default: 0.2, i.e., 20%% of the data)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1000,
        help='Minimum number of training samples required (default: 1000)'
    )
    
    parser.add_argument(
        '--predictor',
        type=str,
        choices=['unified', 'hybrid', 'compare'],
        default='unified',
        help='Predictor to use (unified, hybrid, or compare both)'
    )
    
    parser.add_argument(
        '--by-league',
        action='store_true',
        help='Train and evaluate models separately for each league'
    )
    
    return parser.parse_args()

def plot_comparison(metrics_unified: dict, metrics_hybrid: dict, save_path: Path) -> None:
    """Plot comparison of metrics between unified and hybrid predictors.
    
    Args:
        metrics_unified: Metrics from unified predictor
        metrics_hybrid: Metrics from hybrid predictor
        save_path: Path to save comparison plots
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # For each market, create a comparison plot
    for market in metrics_unified.keys():
        if market not in metrics_hybrid:
            continue
            
        # Get metrics for this market
        unified_metrics = metrics_unified[market]
        hybrid_metrics = metrics_hybrid[market]
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        metrics = list(unified_metrics.keys())
        x = range(len(metrics))
        
        plt.bar([i - 0.2 for i in x], 
                [unified_metrics[m] for m in metrics], 
                width=0.4, 
                label='Unified',
                color='blue',
                alpha=0.6)
        plt.bar([i + 0.2 for i in x], 
                [hybrid_metrics[m] for m in metrics], 
                width=0.4, 
                label='Hybrid',
                color='red',
                alpha=0.6)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{market} - Model Comparison')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path / f'{market}_comparison.png')
        plt.close()

def train_and_evaluate(
    predictor_name: str,
    predictor: UnifiedPredictor | HybridPredictor,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Config,
    args: argparse.Namespace,
    logger: logging.Logger,
    league: str = None
) -> dict:
    """Train and evaluate a predictor.
    
    Args:
        predictor_name: Name of the predictor
        predictor: Predictor instance
        train_data: Training data
        test_data: Test data
        config: Configuration
        args: Command line arguments
        logger: Logger instance
        league: Optional league name for per-league training
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set model directory based on league
    model_dir = config.models_dir / predictor_name
    if league:
        model_dir = model_dir / league
    
    if args.train:
        # Train models
        logger.info("Training models...")
        metrics = predictor.train(
            train_data, 
            save_path=model_dir,
            test_mode=args.test_mode,
            min_samples=args.min_samples
        )
        logger.info(f"{predictor_name} training metrics: %s", metrics)
    else:
        # Load existing models
        logger.info("Loading existing models...")
        predictor.load_models(model_dir)
    
    if args.evaluate:
        # Make predictions on test set
        logger.info("Making predictions...")
        predictions = predictor.predict(test_data)
        
        # Evaluate performance
        logger.info("Evaluating performance...")
        metrics = evaluator.evaluate_predictions(predictions, test_data)
        logger.info(f"{predictor_name} evaluation metrics: %s", metrics)
        
        # Plot results
        output_dir = config.output_dir / predictor_name
        if league:
            output_dir = output_dir / league
        evaluator.plot_performance(save_path=output_dir)
        
        return metrics
    
    return None

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
        evaluator = BettingEvaluator(config.betting)
        
        # Initialize predictors based on argument
        predictors = {}
        if args.predictor in ['unified', 'compare']:
            predictors['unified'] = UnifiedPredictor(config.model)
        if args.predictor in ['hybrid', 'compare']:
            predictors['hybrid'] = HybridPredictor(config.model)
        
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
        
        # Dictionary to store metrics for comparison
        all_metrics = {}
        
        if args.by_league:
            # Train and evaluate separately for each league
            leagues = train_data['League'].unique()
            
            for league in leagues:
                logger.info(f"\nProcessing {league}...")
                
                # Filter data for this league
                league_train = train_data[train_data['League'] == league]
                league_test = test_data[test_data['League'] == league]
                
                logger.info(f"League data size - Train: {len(league_train)}, Test: {len(league_test)}")
                
                league_metrics = {}
                for predictor_name, predictor in predictors.items():
                    metrics = train_and_evaluate(
                        predictor_name,
                        predictor,
                        league_train,
                        league_test,
                        config,
                        args,
                        logger,
                        league
                    )
                    if metrics:
                        league_metrics[predictor_name] = metrics
                
                if league_metrics and args.predictor == 'compare':
                    plot_comparison(
                        league_metrics['unified'],
                        league_metrics['hybrid'],
                        config.output_dir / 'comparison' / league
                    )
        else:
            # Train and evaluate on all data combined
            for predictor_name, predictor in predictors.items():
                metrics = train_and_evaluate(
                    predictor_name,
                    predictor,
                    train_data,
                    test_data,
                    config,
                    args,
                    logger
                )
                if metrics:
                    all_metrics[predictor_name] = metrics
            
            # If comparing, create comparison plots
            if args.predictor == 'compare' and args.evaluate and len(all_metrics) == 2:
                logger.info("Creating comparison plots...")
                plot_comparison(
                    all_metrics['unified'],
                    all_metrics['hybrid'],
                    config.output_dir / 'comparison'
                )
    
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    main() 