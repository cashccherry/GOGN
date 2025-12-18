import numpy as np
import sympy as sp

def derive_D_matrix_diagonal(nx=5, dx=1.0, sigma=1.0, nu=1.0):
    """
    Derive the diagonal elements of D² symbolically for an nx × nx grid.
    D(m) = sigma**-1 * (-L(m) + nu * m) * dx
    Returns a matrix of diagonal coefficients.
    """
    # Create symbolic variables for the grid
    u = sp.symbols('u0:%d' % (nx*nx))
    u_grid = np.array(u).reshape(nx, nx)
    
    # Initialize Laplacian grid
    L = sp.zeros(nx, nx)
    
    # Helper function to get coefficient
    def get_coeff(expr, var):
        """Get coefficient of var in expr"""
        if expr == var:
            return 1
        return expr.coeff(var) if hasattr(expr, 'coeff') else 0
    
    # Define Laplacian (same as in the original code)
    # Interior points
    for i in range(1, nx-1):
        for j in range(1, nx-1):
            L[i,j] = (u_grid[i+1,j] + u_grid[i-1,j] + u_grid[i,j+1] + u_grid[i,j-1] - 4*u_grid[i,j]) / dx**2
    
    # Left boundary (i=0)
    for j in range(1, nx-1):
        L[0,j] = (u_grid[0,j+1] + u_grid[0,j-1] + u_grid[1,j] + u_grid[0,j] - 4*u_grid[0,j]) / dx**2
    
    # Right boundary (i=nx-1)
    for j in range(1, nx-1):
        L[nx-1,j] = (u_grid[nx-1,j+1] + u_grid[nx-1,j-1] + u_grid[nx-1,j] + u_grid[nx-2,j] - 4*u_grid[nx-1,j]) / dx**2
    
    # Bottom boundary (j=0)
    for i in range(1, nx-1):
        L[i,0] = (u_grid[i+1,0] + u_grid[i-1,0] + u_grid[i,1] + u_grid[i,0] - 4*u_grid[i,0]) / dx**2
    
    # Top boundary (j=nx-1)
    for i in range(1, nx-1):
        L[i,nx-1] = (u_grid[i+1,nx-1] + u_grid[i-1,nx-1] + u_grid[i,nx-1] + u_grid[i,nx-2] - 4*u_grid[i,nx-1]) / dx**2
    
    # Corners
    L[0,0] = (u_grid[0,1] + u_grid[0,0] + u_grid[1,0] + u_grid[0,0] - 4*u_grid[0,0]) / dx**2
    L[nx-1,0] = (u_grid[nx-1,1] + u_grid[nx-1,0] + u_grid[nx-1,0] + u_grid[nx-2,0] - 4*u_grid[nx-1,0]) / dx**2
    L[0,nx-1] = (u_grid[0,nx-1] + u_grid[0,nx-2] + u_grid[1,nx-1] + u_grid[0,nx-1] - 4*u_grid[0,nx-1]) / dx**2
    L[nx-1,nx-1] = (u_grid[nx-1,nx-1] + u_grid[nx-1,nx-2] + u_grid[nx-1,nx-1] + u_grid[nx-2,nx-1] - 4*u_grid[nx-1,nx-1]) / dx**2
    
    # Now define D operator
    # D(m) = sigma**-1 * (-L(m) + nu * m) * dx
    D = sp.zeros(nx, nx)
    for i in range(nx):
        for j in range(nx):
            D[i,j] = (1/sigma) * (-L[i,j] + nu * u_grid[i,j]) * dx
    
    # Convert to matrix form for easier computation
    # Flatten the system
    D_flat = sp.zeros(nx*nx, nx*nx)
    for i in range(nx):
        for j in range(nx):
            row_idx = i*nx + j
            for k in range(nx):
                for l in range(nx):
                    col_idx = k*nx + l
                    coeff = get_coeff(D[i,j], u_grid[k,l])
                    if coeff != 0:
                        D_flat[row_idx, col_idx] = coeff
    
    # Compute D² = D_flat * D_flat
    D2_flat = D_flat * D_flat
    
    # Extract diagonal
    diag_vals = []
    for idx in range(nx*nx):
        diag_vals.append(D2_flat[idx, idx])
    
    # Reshape to grid
    diag_grid = np.array(diag_vals).reshape(nx, nx)
    
    return diag_grid, D, L

def verify_D_diagonal_values(nx=5, dx=1.0, sigma=1.0, nu=1.0):
    """Verify the derived diagonal values match our analytical results."""
    diag_grid, D, L = derive_D_matrix_diagonal(nx, dx, sigma, nu)
    
    # Convert sympy expressions to float for comparison
    diag_grid_float = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        for j in range(nx):
            diag_grid_float[i,j] = float(diag_grid[i,j])
    
    # Expected values
    expected = np.zeros((nx, nx))
    
    # We need diag(L) and diag(L²) first
    # Let's compute diag(L) from the Laplacian
    L_diag = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            if i == 0 or i == nx-1:
                if j == 0 or j == nx-1:
                    L_diag[i,j] = -2/dx**2  # Corner
                else:
                    L_diag[i,j] = -3/dx**2  # Boundary
            elif j == 0 or j == nx-1:
                L_diag[i,j] = -3/dx**2  # Boundary
            else:
                L_diag[i,j] = -4/dx**2  # Interior
    
    # diag(L²) from previous derivation
    L2_diag = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            if i == 0 or i == nx-1:
                if j == 0 or j == nx-1:
                    L2_diag[i,j] = 6/dx**4  # Corner
                else:
                    L2_diag[i,j] = 12/dx**4  # Boundary
            elif j == 0 or j == nx-1:
                L2_diag[i,j] = 12/dx**4  # Boundary
            else:
                L2_diag[i,j] = 20/dx**4  # Interior
    
    # Now compute expected diag(D²)
    # D² = (dx/sigma)² * (L² - 2νL + ν²I)
    # diag(D²) = (dx/sigma)² * [diag(L²) - 2ν*diag(L) + ν²]
    factor = (dx/sigma)**2
    
    for i in range(nx):
        for j in range(nx):
            expected[i,j] = factor * (L2_diag[i,j] - 2*nu*L_diag[i,j] + nu**2)
    
    # Compare
    diff = np.abs(diag_grid_float - expected)
    max_diff = np.max(diff)
    
    print(f"Parameters: dx={dx}, sigma={sigma}, nu={nu}")
    print("Symbolically computed D² diagonal values:")
    print(diag_grid_float)
    print("\nExpected D² diagonal values (from formula):")
    print(expected)
    print(f"\nMaximum difference: {max_diff}")
    
    # Also print the simplified formula values
    print("\nSimplified formula check:")
    for i in range(nx):
        for j in range(nx):
            if i == 0 or i == nx-1:
                if j == 0 or j == nx-1:
                    alpha, beta = 6, 4  # Corner
                else:
                    alpha, beta = 12, 6  # Boundary
            elif j == 0 or j == nx-1:
                alpha, beta = 12, 6  # Boundary
            else:
                alpha, beta = 20, 8  # Interior
            
            simplified = (1/sigma**2) * (alpha/dx**2 + beta*nu + nu**2*dx**2)
            symbolic_val = diag_grid_float[i,j]
            print(f"({i},{j}): symbolic={symbolic_val:.6f}, simplified={simplified:.6f}, diff={abs(symbolic_val-simplified):.6e}")
    
    return max_diff < 1e-10

# Test with different parameters
if __name__ == "__main__":
    print("="*60)
    print("Test 1: dx=1.0, sigma=1.0, nu=1.0")
    print("="*60)
    success1 = verify_D_diagonal_values(5, 1.0, 1.0, 1.0)
    
    print("\n" + "="*60)
    print("Test 2: dx=0.5, sigma=2.0, nu=3.0")
    print("="*60)
    success2 = verify_D_diagonal_values(5, 0.5, 2.0, 3.0)
    
    print("\n" + "="*60)
    print("Test 3: dx=0.1, sigma=0.5, nu=10.0")
    print("="*60)
    success3 = verify_D_diagonal_values(5, 0.1, 0.5, 10.0)
    
    print(f"\nAll tests passed: {success1 and success2 and success3}")
