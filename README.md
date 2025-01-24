# Betting Predictor

A machine learning system for predicting sports betting outcomes across multiple markets.

## Features

- Multi-market prediction (match results, over/under, half-time scores, corners, cards)
- Advanced feature engineering with team performance, head-to-head statistics, and form analysis
- Kelly Criterion-based stake sizing
- Market-specific stop-loss limits
- Performance visualization and analysis
- Configurable betting thresholds and risk management

## Project Structure

```
.
├── data/             # Raw and processed data files
├── models/           # Model training and prediction scripts
├── output/           # Performance plots and logs
├── src/
│   ├── analysis/     # Data analysis scripts
│   ├── config/       # Configuration classes
│   ├── data/         # Data loading and preprocessing
│   ├── evaluation/   # Performance evaluation
│   ├── features/     # Feature engineering
│   ├── models/       # Model training and prediction
│   ├── prediction/   # Prediction scripts
│   ├── utils/        # Utility functions
│   ├── visualization/ # Visualization scripts
│   └── main.py       # Main script
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/betting-predictor.git
   cd betting-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your data files in the `data/raw` directory. Files should be in Excel format with the following required columns:
   - Date
   - HomeTeam
   - AwayTeam
   - FTHG (Full-time home goals)
   - FTAG (Full-time away goals)
   - FTR (Full-time result)
   - HTHG (Half-time home goals)
   - HTAG (Half-time away goals)
   - HTR (Half-time result)
   - B365H (Home win odds)
   - B365D (Draw odds)
   - B365A (Away win odds)
   - HC (Home corners)
   - AC (Away corners)
   - HY (Home yellow cards)
   - AY (Away yellow cards)

2. Train models:
   ```bash
   python src/main.py --train
   ```

3. Evaluate performance:
   ```bash
   python src/main.py --evaluate
   ```

4. Train and evaluate:
   ```bash
   python src/main.py --train --evaluate
   ```

## Configuration

The system is highly configurable through the classes in `src/config/config.py`:

- `DataConfig`: Data loading and validation settings
- `FeatureConfig`: Feature engineering parameters
- `ModelConfig`: Model training and market selection
- `BettingConfig`: Betting thresholds and risk management

## Performance Metrics

The system tracks several performance metrics:
- Win rate
- Return on Investment (ROI)
- Profit and Loss (P&L)
- Maximum drawdown
- Market-specific performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 