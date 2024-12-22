"""Visualization module for betting prediction results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Visualizes betting prediction results and model performance."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib using a built-in style
        plt.style.use('seaborn-v0_8')  # Using a specific seaborn version style
        sns.set_theme()  # Apply seaborn theme
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, market: str,
                              top_n: int = 20) -> None:
        """Plot feature importance for a market.
        
        Args:
            importance_df: DataFrame with feature importance
            market: Market name
            top_n: Number of top features to show
        """
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        importance_df = importance_df.head(top_n)
        sns.barplot(data=importance_df, y='feature', x='importance',
                   palette='viridis')
        
        plt.title(f'Top {top_n} Most Important Features - {market.upper()}',
                 pad=20, fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(importance_df['importance']):
            plt.text(v, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{market}_feature_importance.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_pnl_evolution(self, bet_history: pd.DataFrame) -> None:
        """Plot PnL evolution over time.
        
        Args:
            bet_history: DataFrame with betting history
        """
        # Convert date to datetime if needed
        bet_history['date'] = pd.to_datetime(bet_history['date'])
        
        # Calculate cumulative PnL for each strategy and market
        fig = go.Figure()
        
        for market in bet_history['market'].unique():
            for strategy in bet_history['strategy'].unique():
                mask = (bet_history['market'] == market) & \
                       (bet_history['strategy'] == strategy)
                data = bet_history[mask].sort_values('date')
                
                if len(data) > 0:
                    cum_pnl = data['profit'].cumsum()
                    
                    fig.add_trace(go.Scatter(
                        x=data['date'],
                        y=cum_pnl,
                        name=f'{market} - {strategy}',
                        mode='lines',
                        hovertemplate='Date: %{x}<br>PnL: %{y:.2f}<extra></extra>'
                    ))
        
        fig.update_layout(
            title='Cumulative PnL Evolution',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL',
            hovermode='x unified',
            showlegend=True
        )
        
        fig.write_html(self.output_dir / 'pnl_evolution.html')
        
    def plot_win_rate_by_odds(self, bet_history: pd.DataFrame) -> None:
        """Plot win rate distribution by odds ranges.
        
        Args:
            bet_history: DataFrame with betting history
        """
        plt.figure(figsize=(15, 8))
        
        # Create odds ranges
        bet_history['odds_range'] = pd.cut(
            bet_history['odds'],
            bins=[1, 1.5, 2, 2.5, 3, 4, float('inf')],
            labels=['1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '4.0+']
        )
        
        # Calculate win rate for each odds range and strategy
        win_rates = bet_history.groupby(['market', 'strategy', 'odds_range']).\
            agg({'is_winner': ['count', 'sum']})
        win_rates['win_rate'] = win_rates['is_winner']['sum'] / \
                               win_rates['is_winner']['count']
        win_rates = win_rates.reset_index()
        
        # Plot
        g = sns.catplot(
            data=win_rates,
            x='odds_range',
            y='win_rate',
            hue='strategy',
            col='market',
            kind='bar',
            height=6,
            aspect=1.2,
            palette='Set2'
        )
        
        g.set_titles('{col_name}')
        g.set_axis_labels('Odds Range', 'Win Rate')
        
        # Rotate x-labels
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.1%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='bottom'
                )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'win_rates_by_odds.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_roi_by_league(self, bet_history: pd.DataFrame) -> None:
        """Plot ROI distribution by league.
        
        Args:
            bet_history: DataFrame with betting history
        """
        # Check if league information is available
        if 'league' not in bet_history.columns:
            logger.warning("League information not available in betting history. Skipping league ROI plot.")
            return
            
        # Calculate ROI by league
        roi_data = []
        
        for market in bet_history['market'].unique():
            for strategy in bet_history['strategy'].unique():
                market_data = bet_history[
                    (bet_history['market'] == market) &
                    (bet_history['strategy'] == strategy)
                ]
                
                # Group by league and calculate metrics
                league_stats = market_data.groupby('league').agg({
                    'profit': 'sum',
                    'stake': 'count'  # Count of bets
                }).reset_index()
                
                # Calculate ROI for each league
                league_stats['roi'] = league_stats['profit'] / league_stats['stake']
                
                # Add to roi_data
                for _, row in league_stats.iterrows():
                    roi_data.append({
                        'market': market,
                        'strategy': strategy,
                        'league': row['league'],
                        'roi': row['roi'],
                        'bets': row['stake']
                    })
        
        if not roi_data:
            logger.warning("No ROI data available for plotting.")
            return
            
        roi_df = pd.DataFrame(roi_data)
        
        try:
            # Create plot
            fig = px.bar(
                roi_df,
                x='league',
                y='roi',
                color='strategy',
                facet_col='market',
                barmode='group',
                title='ROI by League',
                labels={'roi': 'ROI', 'league': 'League'},
                hover_data=['bets']
            )
            
            fig.update_layout(
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update facet labels
            for annotation in fig.layout.annotations:
                annotation.text = annotation.text.split("=")[1]
            
            fig.write_html(self.output_dir / 'roi_by_league.html')
            logger.info("ROI by league plot saved successfully.")
            
        except Exception as e:
            logger.error(f"Error creating ROI by league plot: {str(e)}")
        
    def plot_confidence_analysis(self, predictions: Dict[str, pd.DataFrame]) -> None:
        """Plot relationship between confidence and accuracy.
        
        Args:
            predictions: Dictionary of prediction DataFrames
        """
        plt.figure(figsize=(15, 10))
        plot_count = 0
        
        for market, preds in predictions.items():
            if 'Confidence' in preds.columns:
                plot_count += 1
                plt.subplot(2, 2, plot_count)
                
                try:
                    # Create confidence bins
                    preds['confidence_bin'] = pd.qcut(
                        preds['Confidence'],
                        q=10,
                        labels=[f'{i*10}-{(i+1)*10}' for i in range(10)]
                    )
                    
                    # Calculate metrics based on confidence bins
                    if market == 'match_result':
                        # For match results, just show confidence distribution
                        accuracy = preds.groupby('confidence_bin').agg({
                            'Confidence': ['count', 'mean']
                        }).reset_index()
                    elif market == 'over_under':
                        # For over/under, show confidence distribution
                        accuracy = preds.groupby('confidence_bin').agg({
                            'Confidence': ['count', 'mean']
                        }).reset_index()
                    else:
                        # For regression markets, show confidence distribution
                        accuracy = preds.groupby('confidence_bin').agg({
                            'Confidence': ['count', 'mean']
                        }).reset_index()
                    
                    # Rename columns for consistent plotting
                    accuracy.columns = ['confidence_bin', 'count', 'confidence']
                    
                    # Plot
                    sns.barplot(
                        data=accuracy,
                        x='confidence_bin',
                        y='confidence',
                        palette='viridis'
                    )
                    
                    plt.title(f'{market} - Confidence Distribution')
                    plt.xlabel('Confidence Range (%)')
                    plt.ylabel('Average Confidence')
                    plt.xticks(rotation=45)
                    
                    # Add count labels
                    for i, row in accuracy.iterrows():
                        plt.text(
                            i,
                            row['confidence'],
                            f'n={int(row["count"])}',
                            ha='center',
                            va='bottom'
                        )
                
                except Exception as e:
                    logger.error(f"Error plotting confidence analysis for {market}: {str(e)}")
                    continue
        
        if plot_count == 0:
            logger.warning("No confidence data available for plotting")
            plt.close()
            return
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Confidence analysis plot saved successfully")
        
    def create_summary_report(self, bet_history: pd.DataFrame,
                            predictions: Dict[str, pd.DataFrame]) -> None:
        """Create HTML summary report with all visualizations.
        
        Args:
            bet_history: DataFrame with betting history
            predictions: Dictionary of prediction DataFrames
        """
        # Create all visualizations
        self.plot_pnl_evolution(bet_history)
        self.plot_win_rate_by_odds(bet_history)
        self.plot_roi_by_league(bet_history)
        self.plot_confidence_analysis(predictions)
        
        # Create summary statistics
        summary_stats = {
            'market': [],
            'strategy': [],
            'total_bets': [],
            'win_rate': [],
            'avg_odds': [],
            'total_pnl': [],
            'roi': []
        }
        
        for market in bet_history['market'].unique():
            for strategy in bet_history['strategy'].unique():
                data = bet_history[
                    (bet_history['market'] == market) &
                    (bet_history['strategy'] == strategy)
                ]
                
                if len(data) > 0:
                    summary_stats['market'].append(market)
                    summary_stats['strategy'].append(strategy)
                    summary_stats['total_bets'].append(len(data))
                    summary_stats['win_rate'].append(data['is_winner'].mean())
                    summary_stats['avg_odds'].append(data['odds'].mean())
                    summary_stats['total_pnl'].append(data['profit'].sum())
                    summary_stats['roi'].append(data['profit'].sum() / len(data))
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Betting Prediction Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .summary {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .visualization {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Betting Prediction Results Summary</h1>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                {summary_df.to_html(index=False, float_format=lambda x: f'{x:.2f}')}
            </div>
            
            <div class="visualization">
                <h2>PnL Evolution</h2>
                <iframe src="pnl_evolution.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            
            <div class="visualization">
                <h2>Win Rates by Odds Range</h2>
                <img src="win_rates_by_odds.png" width="100%">
            </div>
            
            <div class="visualization">
                <h2>ROI by League</h2>
                <iframe src="roi_by_league.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            
            <div class="visualization">
                <h2>Confidence Analysis</h2>
                <img src="confidence_analysis.png" width="100%">
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'summary_report.html', 'w') as f:
            f.write(html_content)
        
        logger.info(f"Summary report generated at {self.output_dir}/summary_report.html") 