"""
baseline_classical.py -- reproduces Table 5.

Applies classical stiff solvers to the INDEX-REDUCED (index-1) forms of both
benchmarks and reports the L2 relative error against the analytic solutions
(B.5) and (B.17). Referenced from Section 4.2 and Table 5.

Also serves as the integrity check on the earlier claims
  "the examples presented in this paper cannot be solved using numerical
   methods -- neither in index-1 nor index-3 forms"  (Sec 4.2)
  "successfully resolves DAE systems that pose challenges for traditional
   numerical methods"                                 (Abstract)
"""
import numpy as np
from scipy.integrate import solve_ivp

print("=" * 72)
print("CHECK 1: Example 1 (particle on circular orbit), index-1 form")
print("=" * 72)

# index-1 form: constraint (B.3) gives lambda explicitly since u1^2+u2^2 != 0
#   0 = z1^2+z2^2+2 u1 u2 (2-u2^2-u1^2) - lambda (u1^2+u2^2)
def lam(u1, u2, z1, z2):
    return (z1**2 + z2**2 + 2*u1*u2*(2 - u2**2 - u1**2)) / (u1**2 + u2**2)

def rhs(t, y):
    u1, u2, z1, z2 = y
    L = lam(u1, u2, z1, z2)
    return [z1, z2, 2*u2 - 2*u2**3 - u1*L, 2*u1 - 2*u1**3 - u2*L]

y0 = [1.0, 0.0, 0.0, 1.0]
for method in ["Radau", "BDF", "LSODA", "DOP853"]:
    sol = solve_ivp(rhs, [0, 1], y0, method=method, rtol=1e-12, atol=1e-14,
                    dense_output=True)
    t = np.linspace(0, 1, 200)
    Y = sol.sol(t)
    ex = np.vstack([np.cos(t), np.sin(t), -np.sin(t), np.cos(t)])
    Lnum = lam(*Y)
    Lex = 1 + np.sin(2*t)
    # L2 relative errors, same metric as the manuscript's RE
    re = [np.linalg.norm(Y[i] - ex[i]) / np.linalg.norm(ex[i]) for i in range(4)]
    reL = np.linalg.norm(Lnum - Lex) / np.linalg.norm(Lex)
    print(f"{method:7s} success={sol.success} nfev={sol.nfev:5d}  "
          f"RE(u1,u2,z1,z2)={['%.2e' % r for r in re]}  RE(lam)={reL:.2e}")

print()
print("Index-3 constraint residual max |u1^2+u2^2-1| along the index-1 solution:")
sol = solve_ivp(rhs, [0, 1], y0, method="Radau", rtol=1e-12, atol=1e-14,
                dense_output=True)
t = np.linspace(0, 1, 2001)
Y = sol.sol(t)
print(f"   {np.max(np.abs(Y[0]**2 + Y[1]**2 - 1)):.3e}")
print("Index-2 constraint residual max |u1 z1 + u2 z2|:")
print(f"   {np.max(np.abs(Y[0]*Y[2] + Y[1]*Y[3])):.3e}")

print()
print("=" * 72)
print("CHECK 2: Example 2 (two-link robot arm), index-1 form")
print("=" * 72)
# M(u) v' = f(u,v,t) - G^T lambda ; index-1 constraint (B.16)
# g_uu: acceleration-level constraint
#   -sin(u1) v1^2 + cos(u1) v1' + (v1'+v2') cos(u1+u2) - sin(u1+u2)(v1+v2)^2 = 0
# G = [cos u1 + cos(u1+u2),  cos(u1+u2)]
def robot(t, Y):
    u1, u2, v1, v2 = Y
    M = np.array([[5 + 3*np.cos(u2), 1 + 1.5*np.cos(u2)],
                  [1 + 1.5*np.cos(u2), 1.0]])
    f = np.array([(np.cos(u1) + np.cos(u1 + u2))*v1 - 3*u1,
                  np.cos(u1 + u2)*v1 + (1 - 1.5*np.cos(u2))*u1])
    G = np.array([np.cos(u1) + np.cos(u1 + u2), np.cos(u1 + u2)])
    # gamma: the part of the acceleration constraint not involving v'
    gamma = -(-np.sin(u1)*v1**2 - np.sin(u1 + u2)*(v1 + v2)**2)
    # solve the index-1 (KKT) saddle system for (v', lambda)
    A = np.zeros((3, 3))
    A[:2, :2] = M
    A[:2, 2] = G
    A[2, :2] = G
    b = np.concatenate([f, [gamma]])
    s = np.linalg.solve(A, b)
    return [v1, v2, s[0], s[1]]

Y0 = [0.0, 0.0, 1.0, -2.0]
for method in ["Radau", "BDF", "LSODA", "DOP853"]:
    sol = solve_ivp(robot, [0, 1], Y0, method=method, rtol=1e-12, atol=1e-14,
                    dense_output=True)
    t = np.linspace(0, 1, 200)
    Y = sol.sol(t)
    ex = np.vstack([np.sin(t), -2*np.sin(t), np.cos(t), -2*np.cos(t)])
    re = [np.linalg.norm(Y[i] - ex[i]) / np.linalg.norm(ex[i]) for i in range(4)]
    print(f"{method:7s} success={sol.success} nfev={sol.nfev:5d}  "
          f"RE(u1,u2,v1,v2)={['%.2e' % r for r in re]}")


# =====================================================================
# Table 5 of the manuscript -- exactly the rows printed below.
# rtol=1e-12, atol=1e-14, 200 uniform test points on [0,1].
# =====================================================================
print()
print("=" * 72)
print("TABLE 5  (manuscript, Section 4.2)")
print("=" * 72)
print(f"{'Example':<11}{'Solver':<12}{'max RE (diff)':>15}{'RE(lambda)':>13}{'nfev':>8}")

t = np.linspace(0, 1, 200)


def _row(name, solver, rhs, Y0, exact, lam_fn=None, lam_exact=None):
    s = solve_ivp(rhs, [0, 1], Y0, method=solver, rtol=1e-12, atol=1e-14,
                  dense_output=True)
    Y = s.sol(t)
    E = exact(t)
    mx = max(np.linalg.norm(Y[i] - E[i]) / np.linalg.norm(E[i])
             for i in range(4))
    if lam_fn is None:
        rl = "---"
    else:
        L, Le = lam_fn(*Y), lam_exact(t)
        rl = f"{np.linalg.norm(L - Le) / np.linalg.norm(Le):.2e}"
    print(f"{name:<11}{solver:<12}{mx:>15.2e}{rl:>13}{s.nfev:>8d}")


ex1_exact = lambda t: np.vstack([np.cos(t), np.sin(t), -np.sin(t), np.cos(t)])
ex2_exact = lambda t: np.vstack([np.sin(t), -2 * np.sin(t),
                                 np.cos(t), -2 * np.cos(t)])

for slv in ("Radau", "BDF"):
    _row("Example 1", slv, rhs, [1.0, 0.0, 0.0, 1.0], ex1_exact,
         lam_fn=lam, lam_exact=lambda t: 1 + np.sin(2 * t))
for slv in ("Radau", "BDF"):
    _row("Example 2", slv, robot, [0.0, 0.0, 1.0, -2.0], ex2_exact)
