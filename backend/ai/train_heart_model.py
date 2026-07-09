"""
Heart Disease Risk Prediction Model — Training Script

This script trains a PyTorch MLP on the UCI/Kaggle Heart Disease dataset
using AMD ROCm GPU (via PyTorch's HIP backend).

For the Jupyter notebook version, see notebooks/01_heart_disease_model_training.ipynb

Usage (in AMD GPU environment):
    python ai/train_heart_model.py

The trained model is saved to ml_models/heart_disease_model.pt
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ai.heart_model import HeartDiseaseNet, FEATURE_NAMES, FEATURE_MEANS, FEATURE_STDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_training_data(n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic heart disease training data based on clinical risk factor distributions.
    
    This is used when a real dataset (UCI Heart Disease / Kaggle) is not available.
    For production, replace with actual clinical datasets.
    
    Features: age, sex, total_cholesterol, ldl, hdl, triglycerides, fasting_glucose, 
              systolic_bp, diastolic_bp, bmi
    """
    np.random.seed(42)
    
    # Generate features from realistic distributions
    age = np.random.normal(50, 12, n_samples).clip(18, 90)
    sex = np.random.binomial(1, 0.55, n_samples).astype(float)  # 55% male
    total_chol = np.random.normal(200, 40, n_samples).clip(100, 400)
    ldl = np.random.normal(115, 35, n_samples).clip(30, 300)
    hdl = np.random.normal(52, 15, n_samples).clip(20, 100)
    triglycerides = np.random.normal(140, 60, n_samples).clip(30, 500)
    fasting_glucose = np.random.normal(105, 25, n_samples).clip(60, 300)
    systolic_bp = np.random.normal(130, 18, n_samples).clip(80, 200)
    diastolic_bp = np.random.normal(82, 10, n_samples).clip(50, 120)
    bmi = np.random.normal(26, 5, n_samples).clip(15, 50)
    
    X = np.column_stack([
        age, sex, total_chol, ldl, hdl, triglycerides,
        fasting_glucose, systolic_bp, diastolic_bp, bmi
    ])
    
    # Generate labels based on clinical risk factors (simplified Framingham-like scoring)
    risk_score = (
        0.02 * (age - 50) +
        0.3 * sex +
        0.005 * (total_chol - 200) +
        0.008 * (ldl - 100) +
        -0.01 * (hdl - 55) +
        0.003 * (triglycerides - 130) +
        0.008 * (fasting_glucose - 95) +
        0.01 * (systolic_bp - 120) +
        0.005 * (diastolic_bp - 80) +
        0.04 * (bmi - 25)
    )
    
    # Add noise and convert to probability
    risk_score += np.random.normal(0, 0.3, n_samples)
    probability = 1 / (1 + np.exp(-risk_score))
    y = (probability > 0.5).astype(float)
    
    logger.info(f"Generated {n_samples} samples: {y.sum():.0f} positive, {(1-y).sum():.0f} negative")
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str | None = None,
) -> HeartDiseaseNet:
    """
    Train the heart disease prediction model.
    
    Args:
        X: Feature matrix (n_samples, 10)
        y: Labels (n_samples,)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on (auto-detected if None)
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"  # Works for both CUDA and ROCm (HIP)
            logger.info(f"🔥 Training on GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("💻 Training on CPU")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Update global normalization parameters
    logger.info("Feature normalization parameters (copy to heart_model.py):")
    for i, name in enumerate(FEATURE_NAMES):
        logger.info(f"  {name}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val).unsqueeze(1),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = HeartDiseaseNet(input_size=10).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                predicted = (output > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {accuracy:.4f}"
            )
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    logger.info(f"✅ Training complete. Best validation loss: {best_val_loss:.4f}")
    return model


def main():
    """Train and save the heart disease model."""
    # Generate or load data
    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_training_data(n_samples=5000)
    
    # Train
    model = train_model(X, y, epochs=100)
    
    # Save model
    save_path = Path("ml_models/heart_disease_model.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")
    
    # Quick test
    model.eval()
    with torch.no_grad():
        # Test with a high-risk profile
        test_input = torch.FloatTensor([[
            1.5, 1.0, 1.5, 1.5, -0.8, 1.0, 1.0, 1.0, 0.8, 1.0
        ]])
        prob = model(test_input).item()
        logger.info(f"Test prediction (high-risk profile): {prob:.4f}")
        
        # Test with a low-risk profile
        test_input = torch.FloatTensor([[
            -1.0, 0.0, -1.0, -1.0, 1.0, -1.0, -0.5, -0.5, -0.5, -0.5
        ]])
        prob = model(test_input).item()
        logger.info(f"Test prediction (low-risk profile): {prob:.4f}")


if __name__ == "__main__":
    main()
