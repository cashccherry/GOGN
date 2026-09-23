"""Reproduce the former scientific-review issues 2 and 3 algebraically.

Uses historical expressions explicitly, not the now-corrected optimizer.
These are small deterministic examples, not paper inversion results.
Run: python tests/diagnose_gogn.py
"""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gogn.objectives.FWI import SimulationPars
from gogn.regularization.totalvariation import TVRegularizer


def diagnose():
    dtype = torch.float64
    # phi_i = (m_i-1)^2/2, R = ||m||^2/2; m_i=1/2 is stationary.
    m = torch.full((2,), 0.5, dtype=dtype)
    rho = torch.sqrt((m-1)**2+2)
    J = torch.diag((m-1)/rho)
    exact_gradient = J.T@rho+m
    code_gradient = J.T@torch.sqrt(rho+1)+m
    gradient_report = dict(model=m.tolist(), exact_gradient=exact_gradient.tolist(),
                           code_gradient=code_gradient.tolist(),
                           code_gradient_norm=float(code_gradient.norm()))

    # TV coefficients match the experiment; only the grid/problem is small.
    torch.manual_seed(42)
    sim = SimulationPars(src=torch.tensor([[96000., 96000.]]),
                         rec=torch.tensor([[192000., 192000.]]), nx=6)
    m = 0.01*torch.randn(36, dtype=dtype)
    reg = TVRegularizer(m0=torch.zeros_like(m), Dx=sim.Dx, Dy=sim.Dy,
                       Dxt=sim.Dxt, Dyt=sim.Dyt, L=sim.laplacian,
                       ε=1e-7, λ=0.4, μ=1e-4, ν=0., h=sim.dx)
    B = torch.stack([reg.hess_vec(m, v) for v in torch.eye(36, dtype=dtype)], dim=1)
    A = torch.randn(3, 36, dtype=dtype)
    data = torch.tensor([0.2, -0.1, 0.3], dtype=dtype)
    residual = A@m-data
    rho = torch.sqrt(residual**2+2)
    J = (residual/rho)[:, None]*A
    gR = reg.grad(m)
    gradient = J.T@rho+gR
    H = B+J.T@J
    K = torch.linalg.solve(B, J.T)
    small = torch.eye(3, dtype=dtype)+J@K
    # This is the active optimizer's expression, isolated from issue 1 and CG error.
    code_direction = K@torch.linalg.solve(small, J@m-rho)-m
    paper_direction = torch.linalg.solve(H, -gradient)
    q = torch.linalg.solve(B, gR)
    general_direction = K@torch.linalg.solve(small, J@q-rho)-q
    torch.testing.assert_close(general_direction, paper_direction)
    torch.testing.assert_close(H@code_direction+gradient, gR-B@m)
    return dict(issue_2=gradient_report, issue_3=dict(
        grid=[6, 6], dx=sim.dx,
        regularizer_gradient_norm=float(gR.norm()),
        relative_regularizer_identity_error=float((gR-B@m).norm()/gR.norm()),
        relative_step_error=float((code_direction-paper_direction).norm()/paper_direction.norm()),
        relative_normal_equation_residual=float((H@code_direction+gradient).norm()/gradient.norm()),
        general_woodbury_relative_residual=float((H@general_direction+gradient).norm()/gradient.norm()),
        minimum_regularizer_eigenvalue=float(torch.linalg.eigvalsh(B).min())))


if __name__ == '__main__':
    print(json.dumps(diagnose(), indent=2))
