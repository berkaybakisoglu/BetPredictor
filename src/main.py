"""Main script for running the betting prediction system."""
import argparse
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
import pandas as pd

from config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from data.loader import DataLoader
from features.engineer import FeatureEngineer
from models.predictor import UnifiedPredictor
from models.hierarchical_predictor import HierarchicalPredictor
from evaluation.evaluator import BettingEvaluator

def setup_logging(output_dir):
    """Setup logging configuration."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, 'run_{0}.log'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run betting prediction system')
    
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--use-hierarchical', action='store_true', 
                      help='Use hierarchical predictor instead of unified predictor')
    parser.add_argument('--compare-models', action='store_true',
                      help='Compare hierarchical and unified predictors')
    
    return parser.parse_args()

def compare_predictors(train_data, test_data, config, logger):
    """Compare hierarchical and unified predictors."""
    logger.info("\nStarting model comparison...")
    
    # Initialize both predictors
    unified_predictor = UnifiedPredictor(config.model)
    hierarchical_predictor = HierarchicalPredictor(config.model)
    
    # Train both predictors
    logger.info("Training Unified Predictor...")
    unified_predictor.train(train_data)
    
    logger.info("Training Hierarchical Predictor...")
    hierarchical_predictor.train(train_data)
    
    # Get predictions from both
    logger.info("Making predictions with both models...")
    unified_predictions = unified_predictor.predict(test_data)
    hierarchical_predictions = hierarchical_predictor.predict(test_data)
    
    # Evaluate both models
    evaluator = BettingEvaluator(config.betting)
    
    logger.info("\nEvaluating Unified Predictor:")
    unified_metrics = evaluator.evaluate(unified_predictions, test_data)
    
    logger.info("\nEvaluating Hierarchical Predictor:")
    hierarchical_metrics = evaluator.evaluate(hierarchical_predictions, test_data)
    
    # Save comparison results
    comparison_dir = os.path.join(config.output_dir, 'model_comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Create comparison visualizations
    plot_model_comparison(unified_metrics, hierarchical_metrics, comparison_dir)
    
    return unified_metrics, hierarchical_metrics

def plot_model_comparison(unified_metrics, hierarchical_metrics, output_dir):
    """Create visualization comparing model performances."""
    # Prepare data for plotting
    metrics_data = []
    
    for market in unified_metrics.keys():
        for metric, value in unified_metrics[market].items():
            if isinstance(value, float):
                metrics_data.append({
                    'Market': market,
                    'Metric': metric,
                    'Value': value,
                    'Model': 'Unified'
                })
        
        for metric, value in hierarchical_metrics[market].items():
            if isinstance(value, float):
                metrics_data.append({
                    'Market': market,
                    'Metric': metric,
                    'Value': value,
                    'Model': 'Hierarchical'
                })
    
    df = pd.DataFrame(metrics_data)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    for i, market in enumerate(unified_metrics.keys(), 1):
        plt.subplot(2, 2, i)
        market_data = df[df['Market'] == market]
        
        sns.barplot(data=market_data, x='Metric', y='Value', hue='Model')
        plt.title('{0} Metrics Comparison'.format(market))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

def main():
    """Main function."""
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
        
        logger.info("Loading data...")
        train_data, test_data = data_loader.load_data(test_mode=args.test_mode)
        
        if args.test_mode:
            logger.info("Running in test mode with reduced dataset...")
            seasons = sorted(train_data['Season'].unique())
            
            if len(seasons) > 2:
                test_seasons = seasons[-2:]
                logger.info("Using only seasons: {0}".format(test_seasons))
                
                train_data = train_data[train_data['Season'].isin(test_seasons)]
                test_data = test_data[test_data['Season'] >= test_seasons[0]]
                
                if args.test_size < 1.0:
                    train_size = int(len(train_data) * args.test_size)
                    test_size = int(len(test_data) * args.test_size)
                    
                    train_data = train_data.sample(n=train_size, random_state=42)
                    test_data = test_data.sample(n=test_size, random_state=42)
                    
                    logger.info("Reduced dataset size - Train: {0}, Test: {1}".format(
                        len(train_data), len(test_data)))
        
        logger.info("Engineering features...")
        train_data = feature_engineer.create_features(train_data, is_training=True)
        test_data = feature_engineer.create_features(test_data, is_training=False)
        
        if args.compare_models:
            logger.info("Comparing predictors...")
            unified_metrics, hierarchical_metrics = compare_predictors(
                train_data, test_data, config, logger
            )
            return
            
        # Choose predictor based on arguments
        predictor = HierarchicalPredictor(config.model) if args.use_hierarchical else UnifiedPredictor(config.model)
        predictor_name = "Hierarchical" if args.use_hierarchical else "Unified"
        
        if args.train:
            logger.info("Training {0} predictor...".format(predictor_name))
            metrics = predictor.train(train_data)
            logger.info("Training metrics: %s", metrics)
            
            # Save model
            predictor.save(config.models_dir)
        else:
            logger.info("Loading existing {0} predictor...".format(predictor_name))
            predictor.load(config.models_dir)
        
        if args.evaluate:
            logger.info("Making predictions...")
            predictions = predictor.predict(test_data)
            
            logger.info("Evaluating predictions...")
            evaluator = BettingEvaluator(config.betting)
            metrics = evaluator.evaluate(predictions, test_data)
            
            logger.info("\nEvaluation metrics:")
            for market, market_metrics in metrics.items():
                logger.info("\n{0}:".format(market.upper()))
                for metric, value in market_metrics.items():
                    if isinstance(value, dict):
                        logger.info("\n- {0}:".format(metric))
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, float):
                                logger.info("  - {0}: {1:.4f}".format(sub_metric, sub_value))
                            else:
                                logger.info("  - {0}: {1}".format(sub_metric, sub_value))
                    elif isinstance(value, float):
                        logger.info("- {0}: {1:.4f}".format(metric, value))
                    else:
                        logger.info("- {0}: {1}".format(metric, value))
            
            # Create visualizations
            if hasattr(evaluator, 'bet_details') and evaluator.bet_details:
                bet_history = pd.DataFrame(evaluator.bet_details)
                visualizer = evaluator.visualizer
                
                visualizer.plot_pnl_evolution(bet_history)
                visualizer.plot_win_rate_by_odds(bet_history)
                visualizer.plot_roi_by_league(bet_history)
                visualizer.plot_confidence_analysis(predictions)
                
                visualizer.create_summary_report(bet_history, predictions)
                evaluator.save_bet_details(os.path.join(config.output_dir, 'betting_results'))
                
                logger.info("Visualizations saved to {0}/visualizations".format(config.output_dir))
            else:
                logger.warning("No betting history available for visualization")
    
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    main() 