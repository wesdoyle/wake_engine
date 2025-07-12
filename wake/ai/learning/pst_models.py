"""
Machine learning models for learning piece-square table values from chess game data.
Implements various approaches from simple regression to neural networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from wake.constants import Piece

@dataclass
class ModelMetrics:
    """Performance metrics for trained models"""
    mse: float
    r2: float
    mae: float
    cross_val_score: float
    training_samples: int

class PSTLearner(ABC):
    """Abstract base class for piece-square table learning models"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """Train the model on feature data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict position values"""
        pass
    
    @abstractmethod
    def get_piece_square_tables(self) -> Dict[int, np.ndarray]:
        """Extract learned piece-square tables"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save trained model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load trained model"""
        pass

class LinearPSTLearner(PSTLearner):
    """Simple linear regression approach to learn piece-square values"""
    
    def __init__(self, regularization: str = 'ridge', alpha: float = 1.0):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LinearPSTLearner")
        
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.piece_coefficients = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """Train linear model"""
        self.model.fit(X, y, sample_weight=sample_weights)
        self._extract_piece_coefficients(X.shape[1])
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict position values"""
        return self.model.predict(X)
    
    def _extract_piece_coefficients(self, n_features: int):
        """Extract piece-specific coefficients from model weights"""
        coeffs = self.model.coef_
        
        # Assuming first 768 features are piece positions (12 pieces * 64 squares)
        piece_order = [Piece.wP, Piece.wN, Piece.wB, Piece.wR, Piece.wQ, Piece.wK,
                      Piece.bP, Piece.bN, Piece.bB, Piece.bR, Piece.bQ, Piece.bK]
        
        for i, piece in enumerate(piece_order):
            start_idx = i * 64
            end_idx = (i + 1) * 64
            if end_idx <= len(coeffs):
                self.piece_coefficients[piece] = coeffs[start_idx:end_idx].reshape(8, 8)
    
    def get_piece_square_tables(self) -> Dict[int, np.ndarray]:
        """Return learned piece-square tables"""
        return self.piece_coefficients.copy()
    
    def save_model(self, filepath: str):
        """Save model using joblib"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load model using joblib"""
        self.model = joblib.load(filepath)

class NeuralPSTLearner(PSTLearner):
    """Neural network approach for learning piece-square values"""
    
    def __init__(self, 
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NeuralPSTLearner")
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self, input_size: int):
        """Build neural network architecture"""
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weights: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 1024, validation_split: float = 0.2):
        """Train neural network"""
        
        if self.model is None:
            self._build_model(X.shape[1])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        if sample_weights is not None:
            weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
        else:
            weights_tensor = None
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=validation_split, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                if weights_tensor is not None:
                    # Apply sample weights (simplified)
                    loss = loss.mean()
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                # Validation loss
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val)
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss.item():.4f}")
                self.model.train()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained neural network"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def get_piece_square_tables(self) -> Dict[int, np.ndarray]:
        """Extract piece-square tables from neural network (approximation)"""
        # This is more complex for neural networks - we need to probe the network
        # with different piece configurations to understand learned values
        
        piece_tables = {}
        piece_order = [Piece.wP, Piece.wN, Piece.wB, Piece.wR, Piece.wQ, Piece.wK,
                      Piece.bP, Piece.bN, Piece.bB, Piece.bR, Piece.bQ, Piece.bK]
        
        for piece in piece_order:
            table = np.zeros((8, 8))
            
            # For each square, create a minimal position with just this piece
            for row in range(8):
                for col in range(8):
                    square = row * 8 + col
                    
                    # Create feature vector with piece on this square only
                    features = np.zeros(768 + 6)  # 768 piece features + 6 other features
                    piece_idx = piece_order.index(piece)
                    features[piece_idx * 64 + square] = 1.0
                    
                    # Predict value
                    value = self.predict(features.reshape(1, -1))[0]
                    table[row, col] = value
            
            piece_tables[piece] = table
        
        return piece_tables
    
    def save_model(self, filepath: str):
        """Save PyTorch model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate
            }, filepath)
    
    def load_model(self, filepath: str):
        """Load PyTorch model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        
        # Rebuild model architecture (input size needs to be known)
        # This is a limitation - we'd need to store input size too
        # For now, assume standard input size
        self._build_model(768 + 6)
        self.model.load_state_dict(checkpoint['model_state_dict'])

class EnsemblePSTLearner(PSTLearner):
    """Ensemble of multiple models for robust learning"""
    
    def __init__(self, models: List[PSTLearner]):
        self.models = models
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """Train all models in ensemble"""
        for model in self.models:
            model.fit(X, y, sample_weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all models"""
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        return np.mean(predictions, axis=0)
    
    def get_piece_square_tables(self) -> Dict[int, np.ndarray]:
        """Average piece-square tables from all models"""
        all_tables = [model.get_piece_square_tables() for model in self.models]
        
        averaged_tables = {}
        piece_types = all_tables[0].keys()
        
        for piece in piece_types:
            tables = [model_tables[piece] for model_tables in all_tables]
            averaged_tables[piece] = np.mean(tables, axis=0)
        
        return averaged_tables
    
    def save_model(self, filepath: str):
        """Save ensemble models"""
        for i, model in enumerate(self.models):
            model.save_model(f"{filepath}_model_{i}")
    
    def load_model(self, filepath: str):
        """Load ensemble models"""
        for i, model in enumerate(self.models):
            model.load_model(f"{filepath}_model_{i}")

class PSTTrainer:
    """Main training pipeline for piece-square table learning"""
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
    
    def train_model(self, 
                   model_name: str,
                   learner: PSTLearner,
                   training_data: Dict[str, np.ndarray],
                   validation_split: float = 0.2) -> ModelMetrics:
        """Train a specific model and return performance metrics"""
        
        X = training_data['features']
        y = training_data['outcomes']
        weights = training_data.get('weights')
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        if weights is not None:
            weights_train = weights[:len(X_train)]
        else:
            weights_train = None
        
        # Train model
        learner.fit(X_train, y_train, weights_train)
        
        # Evaluate
        y_pred = learner.predict(X_test)
        
        metrics = ModelMetrics(
            mse=mean_squared_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
            mae=np.mean(np.abs(y_test - y_pred)),
            cross_val_score=np.mean(cross_val_score(learner.model, X_train, y_train, cv=5)) if hasattr(learner, 'model') else 0.0,
            training_samples=len(X_train)
        )
        
        self.models[model_name] = learner
        self.training_history[model_name] = metrics
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        if not SKLEARN_AVAILABLE:
            # Return simple dict if pandas not available
            return {name: vars(metrics) for name, metrics in self.training_history.items()}
        
        comparison_data = []
        for name, metrics in self.training_history.items():
            comparison_data.append({
                'Model': name,
                'MSE': metrics.mse,
                'R²': metrics.r2,
                'MAE': metrics.mae,
                'CV Score': metrics.cross_val_score,
                'Training Samples': metrics.training_samples
            })
        
        return pd.DataFrame(comparison_data).sort_values('R²', ascending=False)
    
    def get_best_model(self) -> Tuple[str, PSTLearner]:
        """Return the best performing model"""
        best_name = max(self.training_history.keys(), 
                       key=lambda x: self.training_history[x].r2)
        return best_name, self.models[best_name]
    
    def generate_final_tables(self, model_name: Optional[str] = None) -> Dict[int, np.ndarray]:
        """Generate final piece-square tables"""
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        return self.models[model_name].get_piece_square_tables()

# Example usage and model factory
def create_model_ensemble() -> List[PSTLearner]:
    """Create a diverse ensemble of models"""
    models = []
    
    if SKLEARN_AVAILABLE:
        # Linear models
        models.append(LinearPSTLearner(regularization='ridge', alpha=1.0))
        models.append(LinearPSTLearner(regularization='lasso', alpha=0.1))
        
        # Tree-based models (wrapped to fit interface)
        # models.append(RandomForestPSTWrapper())
        # models.append(GradientBoostingPSTWrapper())
    
    if TORCH_AVAILABLE:
        # Neural networks
        models.append(NeuralPSTLearner(hidden_layers=[256, 128, 64]))
        models.append(NeuralPSTLearner(hidden_layers=[512, 256], dropout_rate=0.2))
    
    return models 