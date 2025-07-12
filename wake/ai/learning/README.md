# Data-Driven Piece-Square Table Learning

This module implements a complete machine learning pipeline for learning piece-square table values from chess game databases. Instead of using hand-crafted heuristics, this system analyzes millions of chess games to discover optimal piece placement values.

## 🎯 Overview

Traditional chess engines use static piece-square tables with manually tuned values. This approach learns these values from real game data, considering:

- **Game outcomes**: Positions from winning games vs losing games
- **Player strength**: Weight higher-rated games more heavily  
- **Game phase**: Opening, middlegame, and endgame considerations
- **Positional features**: King safety, piece mobility, pawn structure

## 🏗️ Architecture

### Core Components

1. **Feature Extractor** (`feature_extractor.py`)
   - Extracts comprehensive features from chess positions
   - Handles piece positions, tactical motifs, game context
   - Creates feature vectors for machine learning

2. **Data Pipeline** (`data_pipeline.py`)  
   - Processes PGN files from game databases
   - Filters games by rating, time control, etc.
   - Parallel processing for large datasets

3. **ML Models** (`pst_models.py`)
   - Linear regression with regularization
   - Neural networks for complex patterns
   - Ensemble methods for robust predictions

4. **Training Script** (`train_piece_square_tables.py`)
   - Complete end-to-end pipeline
   - Model comparison and evaluation
   - Automatic table generation

## 🚀 Quick Start

### Prerequisites

```bash
# Required packages
pip install scikit-learn numpy pandas
pip install torch torchvision  # For neural networks (optional)
pip install python-chess      # For PGN processing
```

### Basic Usage

```bash
# Download some game data (example: Lichess database)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2023-12.pgn.bz2
bunzip2 lichess_db_standard_rated_2023-12.pgn.bz2

# Train piece-square tables
python train_piece_square_tables.py \
    --pgn-files lichess_db_standard_rated_2023-12.pgn \
    --model-types linear neural \
    --output-dir ./pst_results
```

### Advanced Configuration

```python
from wake.ai.learning.data_pipeline import ProcessingConfig, PGNProcessor

# Custom processing configuration
config = ProcessingConfig(
    min_elo=1800,           # Minimum player rating
    max_elo=2600,           # Maximum player rating  
    min_moves=30,           # Minimum game length
    sample_interval=5,      # Extract position every 5 moves
    parallel_workers=8,     # CPU cores to use
    filter_time_controls=["rapid", "classical"]  # Time control filters
)

processor = PGNProcessor(config)
```

## 🧠 Machine Learning Approaches

### 1. Linear Models
- **Ridge Regression**: Prevents overfitting with L2 regularization
- **Lasso Regression**: Feature selection with L1 regularization
- **Advantages**: Fast, interpretable, good baseline

### 2. Neural Networks
- **Architecture**: Multi-layer perceptrons with dropout
- **Features**: Handles complex piece interactions
- **Advantages**: Can learn non-linear patterns

### 3. Ensemble Methods
- **Combination**: Multiple models for robust predictions
- **Voting**: Averages predictions from different approaches
- **Advantages**: Better generalization, reduced variance

## 📊 Feature Engineering

### Positional Features
```python
# Example features extracted from each position
features = {
    'piece_positions': {Piece.wP: [8, 9, 10, ...], ...},  # 768 binary features
    'king_safety': [white_safety, black_safety],           # 2 features
    'piece_mobility': {Piece.wN: 4, Piece.bN: 3, ...},   # 12 features  
    'central_control': 2.5,                                # 1 feature
    'pawn_structure': 0.8,                                 # 1 feature
    'game_phase': 0.6,                                     # 1 feature (0=endgame, 1=opening)
}
```

### Game Context Weighting
```python
# Sample weighting based on game quality
weight = rating_weight * phase_weight
rating_weight = min(2.0, avg_rating / 2000)  # Higher rated = more weight
phase_weight = 1.0 + 0.5 * critical_phase    # Middlegame = more weight
```

## 📈 Training Process

### Data Processing Pipeline
1. **Game Filtering**: Remove low-quality games
2. **Position Sampling**: Extract positions at regular intervals  
3. **Feature Extraction**: Convert positions to ML features
4. **Outcome Correlation**: Link positions to game results

### Model Training
1. **Train/Test Split**: 80/20 split for evaluation
2. **Cross-Validation**: 5-fold CV for robust metrics
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Model Selection**: Choose best performing model

### Evaluation Metrics
- **R² Score**: Explained variance in position values
- **MSE**: Mean squared error of predictions
- **MAE**: Mean absolute error
- **Cross-Validation**: Average performance across folds

## 🔧 Integration

### Using Learned Tables

```python
from wake.ai.constants.learned_pst import LEARNED_PIECE_TABLES

class DataDrivenEvaluator:
    def __init__(self):
        self.piece_tables = LEARNED_PIECE_TABLES
    
    def evaluate_position(self, position):
        score = 0
        for piece_type, squares in position.piece_map.items():
            table = self.piece_tables.get(piece_type, [])
            for square in squares:
                row, col = square // 8, square % 8
                if piece_type in Piece.black_pieces:
                    row = 7 - row  # Flip for black
                score += table[row][col]
        return score
```

### A/B Testing

```python
# Compare learned vs traditional tables
traditional_eval = TraditionalEvaluator()
learned_eval = DataDrivenEvaluator()

def evaluate_position_diff(position):
    traditional_score = traditional_eval.evaluate(position)
    learned_score = learned_eval.evaluate(position)
    return learned_score - traditional_score
```

## 📚 Data Sources

### Recommended Databases
1. **Lichess Open Database**
   - URL: https://database.lichess.org/
   - Size: ~100M games per month
   - Format: PGN with ratings, time controls
   - License: Public domain

2. **Chess.com Games**
   - Export personal games
   - High-quality tournament data

3. **FICS/ICC Archives**
   - Historical online games
   - Variety of time controls

### Data Quality Considerations
- **Rating Range**: 1600-2800 ELO for quality play
- **Game Length**: 25-150 moves to avoid short games
- **Time Controls**: Exclude bullet games (too tactical)
- **Completion**: Only finished games with clear outcomes

## 🎛️ Configuration Options

### Processing Configuration
```python
ProcessingConfig(
    min_elo=1600,                    # Minimum player rating
    max_elo=2800,                    # Maximum player rating
    min_moves=25,                    # Minimum game length
    max_moves=150,                   # Maximum game length
    sample_interval=4,               # Extract every Nth position
    max_games_per_file=10000,        # Limit per PGN file
    parallel_workers=4,              # CPU cores
    filter_time_controls=["rapid"],  # Time control filter
    exclude_results=["*"]            # Exclude unfinished games
)
```

### Model Configuration
```python
# Linear models
LinearPSTLearner(regularization="ridge", alpha=1.0)

# Neural networks  
NeuralPSTLearner(
    hidden_layers=[512, 256, 128],
    dropout_rate=0.3,
    learning_rate=0.001
)

# Ensemble
EnsemblePSTLearner([model1, model2, model3])
```

## 🔬 Experimental Results

### Expected Improvements
- **Positional Understanding**: 10-15% better position evaluation
- **Opening Play**: More accurate piece development values  
- **Endgame**: Better king activity and pawn advancement
- **Tactical Awareness**: Improved piece coordination values

### Validation Methods
1. **Engine vs Engine**: Compare learned vs traditional tables
2. **Position Test Suites**: EPD test positions with known evaluations  
3. **Game Outcome Prediction**: Predict winners from positions
4. **Human Expert Validation**: Grandmaster evaluation comparison

## 🚨 Limitations & Considerations

### Current Limitations
- **Computational Cost**: Neural networks require significant training time
- **Data Requirements**: Need millions of high-quality games
- **Feature Engineering**: Manual feature design still required
- **Overfitting Risk**: Models may memorize rather than generalize

### Future Improvements
- **NNUE Integration**: Efficiently updatable neural networks
- **Reinforcement Learning**: Self-play for table refinement
- **Temporal Features**: How piece values change over time
- **Opening-Specific Tables**: Different values per opening

## 🤝 Contributing

### Adding New Features
1. Extend `PositionFeatures` class
2. Update `FeatureExtractor._features_to_vector()`
3. Test with small dataset first
4. Validate feature importance

### Adding New Models
1. Inherit from `PSTLearner` abstract class
2. Implement required methods
3. Add to `create_model_ensemble()`
4. Update training script

### Data Processing
1. Add new filtering criteria to `ProcessingConfig`
2. Extend `_should_process_game()` method
3. Test with sample PGN files
4. Validate output quality

## 📖 References

- **AlphaZero**: Self-play reinforcement learning
- **Stockfish NNUE**: Efficiently updatable neural networks
- **Texel Tuning**: Traditional parameter optimization
- **Deep Blue**: Early computer chess evaluation
- **Lichess Analysis**: Large-scale game analysis

## 📝 License

This piece-square table learning system is part of the Wake chess engine project. The approach combines established machine learning techniques with chess domain knowledge to create data-driven heuristics that improve upon traditional hand-crafted evaluation functions. 