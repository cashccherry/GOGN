## Base Class Tests

import torch
from gogn.regularization.regularizer import Regularizer


class IdentityTestRegularizer(Regularizer):
    def __call__(self, m):
        return 0.5 * torch.dot(m, m)


class QuarticRegularizer(Regularizer):
    def __call__(self, m):
        return torch.sum(m**4)


def test_call():
    reg = Regularizer()

    x = torch.randn(5)

    assert torch.allclose(reg(x), torch.tensor(0.0))


def test_quadratic_grad():
    reg = IdentityTestRegularizer()

    x = torch.randn(10)

    grad = reg.grad(x)

    assert torch.allclose(grad, x)


def test_quadratic_hess_vec():
    reg = IdentityTestRegularizer()

    x = torch.randn(8)
    v = torch.randn(8)

    Hv = reg.hess_vec(x, v)

    assert torch.allclose(Hv, v)


def test_quadratic_hess_solve():
    reg = IdentityTestRegularizer()

    x = torch.randn(10)
    b = torch.randn(10)

    u = reg.hess_solve(x, b)

    assert torch.allclose(u, b, atol=1e-4)


def test_quartic_grad():
    reg = QuarticRegularizer()

    x = torch.randn(12)

    grad = reg.grad(x)

    expected = 4 * x**3

    assert torch.allclose(grad, expected)


def test_quartic_hess_vec():
    reg = QuarticRegularizer()

    x = torch.randn(12)
    v = torch.randn(12)

    Hv = reg.hess_vec(x, v)

    expected = 12 * x**2 * v

    assert torch.allclose(Hv, expected)


def test_quartic_hess_solve():
    reg = QuarticRegularizer()

    # Keep Hessian well-conditioned
    x = torch.randn(12) + 2.0

    b = torch.randn(12)

    u = reg.hess_solve(x, b)

    expected = b / (12 * x**2)

    assert torch.allclose(u, expected, atol=1e-3)


if __name__ == "__main__":
    torch.manual_seed(42)

    tests = [
        test_call,
        test_quadratic_grad,
        test_quadratic_hess_vec,
        test_quadratic_hess_solve,
        test_quartic_grad,
        test_quartic_hess_vec,
        test_quartic_hess_solve,
    ]

    print(f"Running {len(tests)} tests...\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}")
            print(f"    {type(e).__name__}: {e}")
            failed += 1

    print("\n----------------------------------------")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nAll tests passed! 🎉")
    else:
        raise RuntimeError(f"{failed} test(s) failed.")


## Quadratic Regularizer

import torch

from gogn.regularization.quadratic import QuadraticRegularizer


def test_identity_value():
    m = torch.randn(10)
    m0 = torch.randn(10)

    reg = QuadraticRegularizer(
        m0=m0,
        D=lambda x: x,
    )

    expected = 0.5 * torch.linalg.norm(m - m0) ** 2

    assert torch.allclose(reg(m), expected)


def test_identity_grad():
    m = torch.randn(10)
    m0 = torch.randn(10)

    reg = QuadraticRegularizer(
        m0=m0,
        D=lambda x: x,
    )

    expected = m - m0

    assert torch.allclose(reg.grad(m), expected)


def test_identity_hess_vec():
    m = torch.randn(10)
    v = torch.randn(10)

    reg = QuadraticRegularizer(
        m0=torch.zeros_like(m),
        D=lambda x: x,
    )

    assert torch.allclose(reg.hess_vec(m, v), v)


def test_scaled_operator():
    alpha = 3.0

    D = lambda x: alpha * x

    m = torch.randn(8)
    m0 = torch.randn(8)
    v = torch.randn(8)

    reg = QuadraticRegularizer(
        m0=m0,
        D=D,
    )

    expected_grad = alpha * (alpha * (m - m0))
    expected_hv = alpha ** 2 * v

    assert torch.allclose(reg.grad(m), expected_grad)
    assert torch.allclose(reg.hess_vec(m, v), expected_hv)


def test_hess_solve_identity():
    b = torch.randn(12)

    reg = QuadraticRegularizer(
        m0=torch.zeros_like(b),
        D=lambda x: x,
    )

    x = reg.hess_solve(torch.zeros_like(b), b)

    assert torch.allclose(x, b, atol=1e-5)



def test_hess_solve_cholesky():
    n = 10

    A = torch.diag(torch.arange(1, n + 1, dtype=torch.float32))

    D2 = A.T @ A
    D2lo = torch.linalg.cholesky(D2)

    reg = QuadraticRegularizer(
        m0=torch.zeros(n),
        D=lambda x: A @ x,
        Dt=lambda x: A.T @ x,
        D2lo=D2lo,
    )

    b = torch.randn(n)

    x = reg.hess_solve(torch.zeros(n), b)

    expected = torch.linalg.solve(D2, b)

    assert torch.allclose(x, expected, atol=1e-6)

if __name__ == "__main__":
    torch.manual_seed(42)

    tests = [
        test_identity_value,
        test_identity_grad,
        test_identity_hess_vec,
        test_scaled_operator,
        test_hess_solve_identity,
        test_hess_solve_cholesky,
    ]

    print(f"Running {len(tests)} tests...\n")

    passed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}")
            print(f"    {type(e).__name__}: {e}")
            raise

    print(f"\nAll {passed} tests passed! 🎉")
