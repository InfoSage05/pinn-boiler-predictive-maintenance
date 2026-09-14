"""Unit tests for PyTorch PINN model and autograd mechanisms."""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pinn_model import BoilerPINN
from src.physics.preprocessor import BoilerDataPipeline


def test_pinn_forward_and_autograd():
    model = BoilerPINN(in_features=4, hidden_dim=32)
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    
    X_sub = data["X_train"][:10].clone().detach().requires_grad_(True)
    y_sub = data["y_train"][:10]
    deg_sub = data["deg_train"][:10]
    
    # 1. Forward Pass
    temp_pred, foul_pred = model(X_sub)
    assert temp_pred.shape == (10, 1), "Temperature prediction shape mismatch!"
    assert foul_pred.shape == (10, 1), "Fouling prediction shape mismatch!"
    assert torch.all(foul_pred >= 0.0), "Fouling resistance must be strictly non-negative!"
    
    # 2. Gradient Flow & Autograd computation
    loss, metrics = model.compute_losses(X_sub, y_sub, deg_sub, pipeline.scaler_X, pipeline.scaler_y)
    assert loss.item() > 0.0, "Loss must be positive!"
    assert "physics_loss" in metrics, "Physics loss missing from metrics!"
    assert "mono_loss" in metrics, "Monotonicity loss missing from metrics!"
    
    # 3. Backward Pass
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
