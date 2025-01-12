"""Main script for running the betting prediction system."""
import argparse
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
import pandas as pd

from src.config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.predictor import UnifiedPredictor
from src.evaluation.evaluator import BettingEvaluator

def setup_logging(output_dir: Path) -> None:
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
    parser = argparse.ArgumentParser(description='Run betting prediction system')
    
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--test-size', type=float, default=0.2)
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    config = Config(
        data=DataConfig(data_dir=args.data_dir),
        features=FeatureConfig(),
        model=ModelConfig(),
        betting=BettingConfig(),
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
    
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        data_loader = DataLoader(config.data)
        feature_engineer = FeatureEngineer(config.features)
        predictor = UnifiedPredictor(config.model)
        evaluator = BettingEvaluator(config.betting)
        
        logger.info("Loading data...")
        train_data, test_data = data_loader.load_data(test_mode=args.test_mode)
        
        if args.test_mode:
            logger.info("Running in test mode with reduced dataset...")
            seasons = sorted(train_data['Season'].unique())
            
            if len(seasons) > 2:
                # Use last 2 seasons for test mode
                test_seasons = seasons[-2:]
                logger.info(f"Using only seasons: {test_seasons}")
                
                train_data = train_data[train_data['Season'].isin(test_seasons)]
                test_data = test_data[test_data['Season'] >= test_seasons[0]]
                
                if args.test_size < 1.0:
                    train_size = int(len(train_data) * args.test_size)
                    test_size = int(len(test_data) * args.test_size)
                    
                    train_data = train_data.sample(n=train_size, random_state=42)
                    test_data = test_data.sample(n=test_size, random_state=42)
                    
                    logger.info(f"Reduced dataset size - Train: {len(train_data)}, Test: {len(test_data)}")
        
        logger.info("Engineering features...")
        train_data = feature_engineer.create_features(train_data, is_training=True)
        test_data = feature_engineer.create_features(test_data, is_training=False)
        
        if args.train:
            logger.info("Training models...")
            metrics = predictor.train(train_data, save_path=config.models_dir, test_mode=args.test_mode)
            logger.info("Training metrics: %s", metrics)
            
            # Plot feature importance
            logger.info("Plotting feature importance...")
            output_dir = Path("output/feature_importance")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for market in predictor.models:
                importance_df = predictor.feature_importances.get(market)
                if importance_df is not None:
                    plt.figure(figsize=(12, 6))
                    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
                    plt.title(f'Top 20 Most Important Features - {market}')
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{market}_feature_importance.png')
                    plt.close()
                    
                    importance_df.to_csv(output_dir / f'{market}_feature_importance.csv', index=False)
                    
                    logger.info(f"\nTop 10 important features for {market}:")
                    for _, row in importance_df.head(10).iterrows():
                        logger.info(f"{row['feature']}: {row['importance']:.4f}")
        else:
            logger.info("Loading existing models...")
            predictor.load_models(config.models_dir)
        
        if args.evaluate:
            logger.info("Making predictions...")
            predictions = predictor.predict(test_data)
            
            logger.info("Evaluating predictions...")
            metrics = evaluator.evaluate(predictions, test_data)
            
            logger.info("\nEvaluation metrics:")
            for market, market_metrics in metrics.items():
                logger.info(f"\n{market.upper()}:")
                for metric, value in market_metrics.items():
                    if isinstance(value, dict):
                        logger.info(f"\n- {metric}:")
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, float):
                                logger.info(f"  - {sub_metric}: {sub_value:.4f}")
                            else:
                                logger.info(f"  - {sub_metric}: {sub_value}")
                    elif isinstance(value, float):
                        logger.info(f"- {metric}: {value:.4f}")
                    else:
                        logger.info(f"- {metric}: {value}")
            
            logger.info("Creating visualizations...")
            if hasattr(evaluator, 'bet_details') and evaluator.bet_details:
                bet_history = pd.DataFrame(evaluator.bet_details)
                visualizer = evaluator.visualizer
                
                visualizer.plot_pnl_evolution(bet_history)
                visualizer.plot_win_rate_by_odds(bet_history)
                visualizer.plot_roi_by_league(bet_history)
                visualizer.plot_confidence_analysis(predictions)
                
                visualizer.create_summary_report(bet_history, predictions)
                evaluator.save_bet_details(Path(config.output_dir) / 'betting_results')
                
                logger.info(f"Visualizations saved to {config.output_dir}/visualizations")
            else:
                logger.warning("No betting history available for visualization")
    
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    main() 