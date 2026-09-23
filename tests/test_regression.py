"""Small mathematical checks; known scientific discrepancies remain explicit."""
from pathlib import Path
import numpy as np
import pytest
import torch

from gogn.objectives.FWI import SimulationPars, waveformMisfit
from gogn.objectives.utils import parallel_jacobian
from gogn.optimization.nonlinear import GOGN, optParams
from gogn.optimization.utils import line_search_interpolate_grad
from gogn.regularization.quadratic import QuadraticRegularizer
from gogn.regularization.totalvariation import TVRegularizer


def small_sim():
    return SimulationPars(src=torch.tensor([[96000, 96000]]),
                          rec=torch.tensor([[192000, 192000]]), nx=6)


def test_acquisition_and_image_package_data():
    from gogn.objectives.FWI import CWD_file
    root = Path(CWD_file)/'FWI'
    for name, sources in [('good', 8), ('bad', 5), ('good25', 25), ('bad25', 25)]:
        src, rec = [np.load(root/name/f'{kind}.npy') for kind in ('srcs', 'recs')]
        assert src.shape[0] == sources
        assert src.size == sources*2
        assert rec.shape[0] > 0
        assert np.isfinite(src).all() and np.isfinite(rec).all()
    assert small_sim().load_image_as_velocity_field().shape == (36,)


def test_spatial_adjoint_and_laplacian():
    sim = small_sim()
    u, v = torch.randn(36, dtype=torch.float64), torch.randn(36, dtype=torch.float64)
    for D, Dt in [(sim.Dx, sim.Dxt), (sim.Dy, sim.Dyt)]:
        torch.testing.assert_close(torch.dot(D(u), v), torch.dot(u, Dt(v)))
    torch.testing.assert_close(sim.laplacian(u), -sim.Dxt(sim.Dx(u))-sim.Dyt(sim.Dy(u)))
    torch.testing.assert_close(sim.laplacian(torch.ones_like(u)), torch.zeros_like(u))


def test_tv_derivatives_against_autograd():
    sim = small_sim()
    m = torch.randn(36, dtype=torch.float64, requires_grad=True)*0.01
    v = torch.randn_like(m)
    reg = TVRegularizer(m0=torch.zeros_like(m), Dx=sim.Dx, Dy=sim.Dy,
                       Dxt=sim.Dxt, Dyt=sim.Dyt, L=sim.laplacian,
                       ε=1e-7, λ=0.4, μ=1e-4, ν=0., h=sim.dx)
    grad, = torch.autograd.grad(reg(m), m, create_graph=True)
    hv, = torch.autograd.grad(grad, m, grad_outputs=v)
    torch.testing.assert_close(reg.grad(m), grad)
    torch.testing.assert_close(reg.hess_vec(m, v), hv)


def test_source_residual_gradient_and_counts():
    m = torch.tensor([0.2, -0.3], requires_grad=True)
    fs = [waveformMisfit(lambda x, i=i: x[i:i+1], torch.ones(1), torch.ones(1)) for i in range(2)]
    jac, res = parallel_jacobian(fs, m)
    torch.testing.assert_close(jac.T@res, m.detach()-1)
    assert [(f.forward_counter, f.backward_counter) for f in fs] == [(1, 1), (1, 1)]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('matrix_rhs', [False, True])
def test_quadratic_nondiagonal_inverse(dtype, matrix_rhs):
    D = torch.tensor([[2., 0.], [1., 3.]], dtype=dtype)
    H = D.T@D
    reg = QuadraticRegularizer(m0=torch.zeros(2), D=lambda x: D@x,
                              Dt=lambda x: D.T@x, D2lo=torch.linalg.cholesky(H))
    b = torch.tensor([1., -2.], dtype=dtype)
    if matrix_rhs:
        b = torch.stack([b, b.flip(0)], dim=1)
    solution = reg.hess_solve(torch.zeros(2, dtype=dtype), b)
    torch.testing.assert_close(solution, torch.linalg.solve(H, b))
    torch.testing.assert_close(H @ solution, b)


def test_gogn_recorded_gradient_matches_objective():
    initial = torch.zeros(2, requires_grad=True)
    fs = [waveformMisfit(lambda x, i=i: x[i:i+1], torch.ones(1), torch.ones(1)) for i in range(2)]
    reg = QuadraticRegularizer(m0=initial.detach(), D=lambda x: x, D2lo=torch.eye(2))
    result, history = GOGN(fs, initial, optParams(maxiter=2, maxstep=0.1, targetx=torch.ones(2)),
                           reg=reg, print_debug=False, line_search=line_search_interpolate_grad)
    expected = torch.linalg.vector_norm(2*result.detach()-1)
    torch.testing.assert_close(torch.tensor(history.gradnorm[-1]), expected)


def test_quadratic_inverse_gradient_uses_no_operator_or_solve(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Quadratic shortcut must not evaluate D, grad, or a solve')
    m0 = torch.tensor([0.2, -0.4])
    m = torch.tensor([0.7, 0.1], requires_grad=True)
    reg = QuadraticRegularizer(m0=m0, D=forbidden)
    monkeypatch.setattr(reg, 'grad', forbidden)
    monkeypatch.setattr(reg, 'hess_solve', forbidden)
    torch.testing.assert_close(reg.hess_solve_grad(m), m.detach()-m0)


def test_tv_gogn_direction_matches_normal_equation(monkeypatch):
    sim = small_sim()
    initial = (0.01*torch.randn(36)).requires_grad_()
    reg = TVRegularizer(m0=torch.zeros_like(initial), Dx=sim.Dx, Dy=sim.Dy,
                       Dxt=sim.Dxt, Dyt=sim.Dyt, L=sim.laplacian,
                       ε=1e-7, λ=0.4, μ=1e-4, ν=0., h=sim.dx)
    A, data = torch.randn(3, 36), torch.tensor([0.2, -0.1, 0.3])
    fs = [waveformMisfit(lambda x, i=i: (A@x)[i:i+1], data[i:i+1], torch.ones(1))
          for i in range(3)]
    residual = A@initial.detach()-data
    rho = torch.sqrt(residual**2+2)
    J = (residual/rho)[:, None]*A
    B = torch.stack([reg.hess_vec(initial.detach(), v) for v in torch.eye(36)], dim=1)
    gradient = J.T@rho+reg.grad(initial.detach())
    expected = torch.linalg.solve(B+J.T@J, -gradient)
    # Check the actual TV iterative inverse separately, then isolate the
    # optimizer algebra with exact solves (independent of its CG tolerance).
    q = reg.hess_solve_grad(initial)
    assert torch.linalg.vector_norm(B@q-reg.grad(initial.detach())) <= 0.0051
    monkeypatch.setattr(reg, 'hess_solve', lambda m, b: torch.linalg.solve(B, b))
    captured = []
    def fixed_step(f, x, p, *args, **kwargs):
        captured.append(p.detach().clone())
        trial = x + 0.001*p
        return 0.001, f(trial), trial
    GOGN(fs, initial, optParams(maxiter=2, maxstep=1., targetx=torch.zeros(36)),
         reg=reg, print_debug=False, line_search=fixed_step)
    torch.testing.assert_close(captured[0], expected, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close((B+J.T@J)@captured[0], -gradient, rtol=1e-4, atol=2e-5)
