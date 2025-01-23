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
    
    parser.add_argument('--data-dir', type=str, default='data/raw_csv')
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
    
    unified_predictor = UnifiedPredictor(config.model)
    hierarchical_predictor = HierarchicalPredictor(config.model)
    
    logger.info("Training Unified Predictor...")
    unified_results = unified_predictor.train(train_data)
    
    logger.info("Training Hierarchical Predictor...")
    hierarchical_results = hierarchical_predictor.train(train_data)
    
    logger.info("Making predictions with both models...")
    unified_predictions = unified_predictor.predict(test_data)
    hierarchical_predictions = hierarchical_predictor.predict(test_data)
    
    evaluator = BettingEvaluator(config.betting)
    
    logger.info("\nEvaluating Unified Predictor:")
    unified_metrics = evaluator.evaluate(unified_predictions, test_data)
    
    logger.info("\nEvaluating Hierarchical Predictor:")
    hierarchical_metrics = evaluator.evaluate(hierarchical_predictions, test_data)
    
    # Save feature importance plots
    comparison_dir = os.path.join(config.output_dir, 'model_comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    hierarchical_predictor.plot_feature_importance(
        os.path.join(comparison_dir, 'hierarchical_features')
    )
    
    # Create comparison visualizations
    plot_model_comparison(
        unified_metrics, hierarchical_metrics, 
        unified_results['metrics'], hierarchical_results['metrics'],
        comparison_dir
    )
    
    return unified_metrics, hierarchical_metrics

def plot_model_comparison(unified_metrics, hierarchical_metrics, 
                         unified_training, hierarchical_training, output_dir):
    """Create visualization comparing model performances."""
    metrics_data = []
    
    # Betting performance metrics
    for market in unified_metrics.keys():
        if market == 'bankroll':
            continue
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
    
    # Plot betting metrics
    plt.figure(figsize=(15, 10))
    
    for i, market in enumerate(unified_metrics.keys()):
        if market == 'bankroll':
            continue
        plt.subplot(2, 2, i)
        market_data = df[df['Market'] == market]
        
        sns.barplot(data=market_data, x='Metric', y='Value', hue='Model')
        plt.title(f'{market} Metrics Comparison')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'betting_metrics_comparison.png'))
    plt.close()
    
    # Plot bankroll evolution
    plt.figure(figsize=(12, 6))
    plt.plot(unified_metrics['bankroll']['balance_history'], 
             label='Unified', alpha=0.7)
    plt.plot(hierarchical_metrics['bankroll']['balance_history'], 
             label='Hierarchical', alpha=0.7)
    plt.title('Bankroll Evolution')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'bankroll_evolution.png'))
    plt.close()
    
    # Plot training metrics comparison
    training_data = []
    for market in unified_training.keys():
        for metric, value in unified_training[market].items():
            if isinstance(value, float):
                training_data.append({
                    'Market': market,
                    'Metric': metric,
                    'Value': value,
                    'Model': 'Unified'
                })
        
        for metric, value in hierarchical_training[market].items():
            if isinstance(value, float):
                training_data.append({
                    'Market': market,
                    'Metric': metric,
                    'Value': value,
                    'Model': 'Hierarchical'
                })
    
    training_df = pd.DataFrame(training_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=training_df, x='Market', y='Value', hue='Model')
    plt.title('Training Metrics Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics_comparison.png'))
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create base config
    config = Config()
    
    # Only override data_dir if explicitly provided
    if args.data_dir != 'data/raw_csv/data':  # Only override if different from default
        config.data.data_dir = args.data_dir
    
    # Override other settings
    config.output_dir = args.output_dir
    config.models_dir = args.models_dir
    
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        data_loader = DataLoader(config.data)
        feature_engineer = FeatureEngineer(config.features)
        
        logger.info("Loading data...")
        train_data, test_data = data_loader.load_data(test_mode=args.test_mode)
        
        logger.info("Engineering features...")
        train_data = feature_engineer.create_features(train_data, is_training=True)
        test_data = feature_engineer.create_features(test_data, historical_data=train_data, is_training=False)
        
        if args.compare_models:
            logger.info("Comparing predictors...")
            unified_metrics, hierarchical_metrics = compare_predictors(
                train_data, test_data, config, logger
            )
            return
        
        predictor = HierarchicalPredictor(config.model) if args.use_hierarchical else UnifiedPredictor(config.model)
        predictor_name = "Hierarchical" if args.use_hierarchical else "Unified"
        
        if args.train:
            logger.info("Training {0} predictor...".format(predictor_name))
            training_results = predictor.train(train_data)
            
            logger.info("\nTraining metrics:")
            for market, metrics in training_results['metrics'].items():
                logger.info(f"\n{market.upper()}:")
                for metric, value in metrics.items():
                    logger.info(f"- {metric}: {value:.4f}")
            
            # Plot feature importance
            predictor.plot_feature_importance(
                os.path.join(config.output_dir, 'feature_importance')
            )
            
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
                if market == 'bankroll':
                    logger.info("\nBankroll Metrics:")
                    logger.info(f"Final Balance: {market_metrics['final_balance']:.2f}")
                    logger.info(f"Max Drawdown: {market_metrics['max_drawdown']:.2%}")
                    logger.info(f"Sharpe Ratio: {market_metrics['sharpe_ratio']:.2f}")
                    logger.info(f"Profit Factor: {market_metrics['profit_factor']:.2f}")
                    continue
                    
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
            
            # Create visualizations directory
            viz_dir = os.path.join(config.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Plot feature importance if available
            if hasattr(predictor, 'plot_feature_importance'):
                predictor.plot_feature_importance(viz_dir)
            
            logger.info(f"\nVisualizations saved to {viz_dir}")
            
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