import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path


MMHG_PER_DYN_CM2 = 1.0 / 1333.22


class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class PinnStenosis:
    def __init__(self, model_dir="models/pinn", device="cpu"):
        self.model_dir = model_dir
        self.device = device

        d = Path(self.model_dir)

        self.scaler_X = joblib.load(d / "scaler_X_pinn.pkl")
        self.scaler_y = joblib.load(d / "scaler_y_pinn.pkl")

        self.model = SimplePINN().to(self.device)

        state = torch.load(
            d / "pinn_model.pth",
            map_location=self.device
        )

        self.model.load_state_dict(state)
        self.model.eval()

    def dp_norm_from_Re(self, Re, Lr, Ds, asym=0.0):

        log_re = np.log(float(Re))

        features = np.array(
            [
                log_re,
                float(Lr),
                float(Ds),
                float(asym),
            ],
            dtype=np.float32
        ).reshape(1, -1)

        Xs = self.scaler_X.transform(features).astype(np.float32)

        with torch.no_grad():
            x_tensor = torch.from_numpy(Xs).to(self.device)
            y_scaled = self.model(x_tensor).cpu().numpy()

        y_log = self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        )

        dp_norm = np.expm1(y_log).squeeze()

        return float(dp_norm)

    def dp_mmhg_from_Re(
        self,
        Re,
        Lr,
        Ds,
        rho_phys,
        mu_phys,
        R_ref,
        asym=0.0,
        signed_by_Q=None,
    ):


        dp_norm = self.dp_norm_from_Re(
            Re,
            Lr,
            Ds,
            asym
        )

        V_ref = (
            float(Re)
            * float(mu_phys)
            / (2.0 * float(rho_phys) * float(R_ref))
        )

        dp_mmhg = (
            dp_norm
            * float(rho_phys)
            * V_ref
            * V_ref
            * MMHG_PER_DYN_CM2
        )

        if signed_by_Q is not None:
            dp_mmhg = np.sign(float(signed_by_Q)) * abs(dp_mmhg)

        return float(dp_mmhg)


_PINN_SINGLETON = None


def get_pinn(model_dir="models/pinn", device="cpu"):
    global _PINN_SINGLETON

    if _PINN_SINGLETON is None:
        _PINN_SINGLETON = PinnStenosis(
            model_dir=model_dir,
            device=device
        )

    return _PINN_SINGLETON