"""
Pure PyTorch Physics-Informed Neural Network (PINN) for Industrial Boilers.
Features:
- Dual-head architecture: Forward State (T_supply) + Inverse Degradation (R_fouling)
- Algorithmic Automatic Differentiation via torch.autograd.grad
- 1st Law Energy Balance residual loss
- Physical Monotonicity prior for fouling accumulation
- Thermodynamic boundary constraint enforcement
"""

import time
import numpy as np
import torch
import torch.nn as nn
from src.physics.boiler_thermo import BoilerThermodynamics


class BoilerPINN(nn.Module):
    def __init__(
        self,
        in_features=4,
        hidden_dim=64,
        w_data=1.0,
        w_phys=0.15,
        w_mono=0.05,
        w_bound=0.02,
        w_inverse=0.50
    ):
        super().__init__()
        self.thermo = BoilerThermodynamics()
        self.w_data = w_data
        self.w_phys = w_phys
        self.w_mono = w_mono
        self.w_bound = w_bound
        self.w_inverse = w_inverse
        self.name = "Physics-Informed Neural Network (PINN)"
        
        # Shared trunk using Tanh for smooth second-order derivatives in autograd
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Forward state prediction head (Supply Temperature)
        self.head_temp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Inverse degradation parameter head (Fouling Resistance Rf >= 0)
        self.head_fouling = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus() # Guarantees non-negative physical resistance Rf >= 0
        )

    def forward(self, x):
        """
        Forward pass returning both temperature and latent fouling resistance.
        x: [batch, in_features]
        """
        features = self.shared_trunk(x)
        temp_pred = self.head_temp(features)
        foul_pred = self.head_fouling(features)
        return temp_pred, foul_pred

    def compute_losses(
        self,
        x_scaled,
        y_scaled,
        deg_true,
        scaler_X,
        scaler_y
    ):
        """
        Calculates composite loss with explicit torch.autograd gradient computation:
        L_total = w_data * L_data + w_phys * L_phys + w_mono * L_mono + w_bound * L_bound + w_inv * L_inv
        """
        # Ensure input requires gradients for autograd differential operators
        x_req = x_scaled.clone().detach().requires_grad_(True)
        
        # Forward pass
        temp_scaled_pred, foul_pred = self(x_req)
        
        # 1. Data Matching Loss (Normalized MSE)
        l_data = torch.mean((temp_scaled_pred - y_scaled) ** 2)
        
        # 2. Inverse Degradation Identification Loss
        l_inverse = torch.mean((foul_pred - deg_true) ** 2)
        
        # 3. Autograd Physics Energy Balance Loss
        # Unscale inputs to physical units for thermodynamic calculation
        x_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=x_scaled.device)
        x_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=x_scaled.device)
        y_mean = float(scaler_y.mean_[0])
        y_scale = float(scaler_y.scale_[0])
        
        x_physical = x_req * x_scale + x_mean
        t_supply_physical = temp_scaled_pred * y_scale + y_mean
        
        fuel_mdot = x_physical[:, 0:1]
        t_air = x_physical[:, 1:2]
        t_return = x_physical[:, 2:3]
        water_mdot = x_physical[:, 3:4]
        
        # Compute gradient of predicted temperature w.r.t input features via autograd
        grad_t = torch.autograd.grad(
            outputs=temp_scaled_pred,
            inputs=x_req,
            grad_outputs=torch.ones_like(temp_scaled_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # In thermodynamics, dT/d(water_mdot) MUST be strictly negative (more flow -> lower outlet temp)
        # We penalize unphysical positive derivatives w.r.t water flow!
        grad_water_flow = grad_t[:, 3:4]
        l_mono = torch.mean(torch.relu(grad_water_flow)) # Penalizes dT/dm_w > 0
        
        # 1st Law steady-state / quasi-static heat balance residual (kW):
        # Q_eff = m_w * cp * (T_supply - T_return)
        # 1/Q_eff = 1/Q_clean + gamma * R_foul
        q_clean = self.thermo.q_clean_kw * (0.98 + 0.008 * fuel_mdot + 0.0002 * (t_air - 293.15))
        inv_q_eff = (1.0 / q_clean) + self.thermo.gamma_foul * foul_pred
        q_eff_predicted = 1.0 / torch.clamp(inv_q_eff, min=1e-5)
        
        q_absorbed_water = water_mdot * self.thermo.water_cp * (t_supply_physical - t_return)
        energy_imbalance = q_absorbed_water - q_eff_predicted
        l_physics = torch.mean((energy_imbalance / 50.0) ** 2) # Normalized residual penalty
        
        # 4. Thermodynamic Boundary Loss (T_supply >= T_return)
        l_boundary = torch.mean(torch.relu(t_return - t_supply_physical))
        
        total_loss = (
            self.w_data * l_data
            + self.w_phys * l_physics
            + self.w_mono * l_mono
            + self.w_bound * l_boundary
            + self.w_inverse * l_inverse
        )
        
        metrics = {
            "total_loss": float(total_loss.item()),
            "data_loss": float(l_data.item()),
            "physics_loss": float(l_physics.item()),
            "mono_loss": float(l_mono.item()),
            "inverse_loss": float(l_inverse.item())
        }
        return total_loss, metrics

    def fit_pinn(
        self,
        X_train,
        y_train,
        deg_train,
        scaler_X,
        scaler_y,
        epochs=350,
        lr=0.003,
        batch_size=256,
        verbose=True
    ):
        """Trains PINN with Adam and autograd physics backpropagation."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train, deg_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {"total": [], "data": [], "physics": [], "mono": [], "inverse": []}
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.train()
            batch_metrics = {"total": 0.0, "data": 0.0, "physics": 0.0, "mono": 0.0, "inverse": 0.0}
            n_batches = 0
            
            for bx, by, bdeg in loader:
                optimizer.zero_grad()
                loss, m = self.compute_losses(bx, by, bdeg, scaler_X, scaler_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                optimizer.step()
                
                for k in batch_metrics:
                    batch_metrics[k] += m[f"{k}_loss"]
                n_batches += 1
                
            scheduler.step()
            
            for k in history:
                history[k].append(batch_metrics[k] / n_batches)
                
            if verbose and (epoch % 50 == 0 or epoch == epochs):
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch:03d}/{epochs}] "
                    f"Total: {history['total'][-1]:.4f} | "
                    f"Data: {history['data'][-1]:.4f} | "
                    f"Phys: {history['physics'][-1]:.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
        return history

    def predict_with_diagnostics(self, X_scaled, scaler_X, scaler_y):
        """
        Runs inference and computes physical consistency diagnostics.
        Returns:
            t_pred_physical: Predicted outlet temperature in Kelvin
            foul_pred: Estimated unobservable fouling resistance
            physics_residuals: Energy balance residual in kW
        """
        self.eval()
        with torch.no_grad():
            t_scaled_pred, foul_pred = self(X_scaled)
            
        t_pred_np = scaler_y.inverse_transform(t_scaled_pred.cpu().numpy())
        foul_pred_np = foul_pred.cpu().numpy()
        X_physical = scaler_X.inverse_transform(X_scaled.cpu().numpy())
        
        fuel_mdot = X_physical[:, 0:1]
        t_air = X_physical[:, 1:2]
        t_return = X_physical[:, 2:3]
        water_mdot = X_physical[:, 3:4]
        
        # Calculate physical residual in kW
        q_clean = self.thermo.q_clean_kw * (0.98 + 0.008 * fuel_mdot + 0.0002 * (t_air - 293.15))
        inv_q_eff = (1.0 / q_clean) + self.thermo.gamma_foul * np.maximum(foul_pred_np, 0.0)
        q_eff = 1.0 / np.maximum(inv_q_eff, 1e-5)
        
        q_absorbed = water_mdot * self.thermo.water_cp * (t_pred_np - t_return)
        residuals_kw = np.abs(q_absorbed - q_eff)
        
        return {
            "t_pred_k": t_pred_np,
            "fouling_pred": foul_pred_np,
            "physics_residual_kw": residuals_kw
        }
