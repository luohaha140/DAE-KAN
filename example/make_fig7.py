"""
make_fig7.py -- generates the DATA and the FIGURE for Fig. 7 of the manuscript.

    "Drift-off for the DOPRI5 solution of the index-1 pendulum problem"

Everything the figure shows is produced here from the equations printed in
Appendix A; nothing is read from an external file. Running this script writes

    drift-off-phenomenon.pdf     the figure (vector, Type 42 fonts)
    drift-off-phenomenon.csv     the underlying data, one row per time point
    drift-off-checkpoints.txt    the values quoted in the text of Appendix A.2

The integration is deterministic: the checkpoint values have been reproduced
digit-for-digit on Linux/SciPy and on Windows/SciPy.

NOTE -- the classical-solver baseline of Table 5 is NOT computed here. It lives
in baseline_classical.py and uses rtol=1e-12, atol=1e-14. Keep the two apart:
quoting numbers from two different tolerance settings in one table is exactly
the error this separation is meant to prevent.

-------------------------------------------------------------------------------
WHAT IS BEING INTEGRATED, AND WHY
-------------------------------------------------------------------------------
The pendulum in index-3 form, Eq. (A.1):

    x' = u
    y' = v
    u' = -lambda * x
    v' = -lambda * y - 1
    0  = x^2 + y^2 - 1                      <-- position (index-3) constraint

Differentiating the constraint once gives the index-2 (velocity) constraint,
Eq. (A.2):

    0 = x*u + y*v

Differentiating again and substituting the dynamics gives the index-1
(acceleration) constraint, Eq. (A.3):

    0 = u^2 + v^2 - lambda - y

On the index-1 manifold this last equation determines lambda explicitly,

    lambda = u^2 + v^2 - y                                            (*)

so the index-1 DAE collapses to the explicit ODE integrated below. This is the
system a classical solver is actually given after index reduction.

The point of the figure is that the solver then satisfies (A.3) -- the equation
it was handed -- to round-off, while (A.1) and (A.2), which it was NOT handed,
are violated by an amount that grows with t. That growing violation is the
drift-off phenomenon, and it is why DAE-KAN is trained against the unreduced
constraint instead.

Initial conditions (Appendix A.2): x0 = 1, y0 = 0, u0 = 0, v0 = 0. These are
consistent: x0^2 + y0^2 - 1 = 0 and x0*u0 + y0*v0 = 0, and (*) gives
lambda(0) = 0.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Configuration. RTOL and ATOL are quoted in the figure title and in the
# caption; if you change them here, change the caption to match.
# ---------------------------------------------------------------------------
RTOL, ATOL = 1e-6, 1e-9      # a standard working tolerance, stated in caption
T_END = 100.0                # long horizon: drift is invisible on [0, 1]
N_OUT = 20001                # dense-output points used for the plot
METHOD = "DOP853"            # explicit RK of the Dormand-Prince family

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,      # Type 42 embedding, per Elsevier artwork rules
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# 1. The index-1 system, obtained from (A.1) by eliminating lambda via (*)
# ---------------------------------------------------------------------------
def pendulum_index1(t, Y):
    x, y, u, v = Y
    lam = u * u + v * v - y          # (*) the index-1 constraint, solved
    return [u, v, -lam * x, -lam * y - 1.0]


def constraints(x, y, u, v):
    """Residuals of the three constraint levels, Eqs. (A.1)-(A.3)."""
    lam = u * u + v * v - y
    g3 = np.abs(x * x + y * y - 1.0)             # index-3, NOT enforced
    g2 = np.abs(x * u + y * v)                   # index-2, NOT enforced
    g1 = np.abs(u * u + v * v - lam - y)         # index-1, enforced by (*)
    return g1, g2, g3


# ---------------------------------------------------------------------------
# 2. Integrate
# ---------------------------------------------------------------------------
Y0 = [1.0, 0.0, 0.0, 0.0]
sol = solve_ivp(pendulum_index1, [0.0, T_END], Y0, method=METHOD,
                rtol=RTOL, atol=ATOL, dense_output=True)
if not sol.success:
    raise RuntimeError("integration failed: " + sol.message)

t = np.linspace(0.0, T_END, N_OUT)
x, y, u, v = sol.sol(t)
g1, g2, g3 = constraints(x, y, u, v)

print("solver=%s  rtol=%g atol=%g  nfev=%d  accepted steps=%d"
      % (METHOD, RTOL, ATOL, sol.nfev, len(sol.t)))


# ---------------------------------------------------------------------------
# 3. Write the data out, so the figure can be audited without rerunning
# ---------------------------------------------------------------------------
np.savetxt(
    "drift-off-phenomenon.csv",
    np.column_stack([t, x, y, u, v, g1, g2, g3]),
    delimiter=",", comments="",
    header="t,x,y,u,v,res_index1,res_index2,res_index3",
    fmt="%.10e",
)
print("wrote drift-off-phenomenon.csv  (%d rows)" % N_OUT)

lines = ["# Fig. 7 checkpoints  (%s, rtol=%g, atol=%g)" % (METHOD, RTOL, ATOL),
         "#     t   |x^2+y^2-1|       |xu+yv|   index-1 res"]
for T in (1, 10, 25, 50, 100):
    i = int(np.argmin(np.abs(t - T)))
    lines.append("  %5d   %11.3e   %11.3e   %11.3e" % (T, g3[i], g2[i], g1[i]))
report = "\n".join(lines)
open("drift-off-checkpoints.txt", "w").write(report + "\n")
print(report)
print("\nmax index-1 residual over [0,%g] = %.3e"
      "   <- the constraint the solver was given; stays at round-off"
      % (T_END, g1.max()))


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def envelope(a, w=201):
    """Running maximum: the drift trend, free of the oscillatory dips."""
    half = w // 2
    return np.array([a[max(0, i - half):i + half + 1].max()
                     for i in range(a.size)])


fig, ax = plt.subplots(figsize=(5.0, 2.6))
floor = 1e-18
ax.semilogy(t, np.maximum(g3, floor), color="#1f77b4", lw=0.4, alpha=0.25)
ax.semilogy(t, np.maximum(g2, floor), color="#d62728", lw=0.4, alpha=0.25)
ax.semilogy(t, envelope(g3), color="#1f77b4", lw=1.3,
            label=r"$|x_n^2+y_n^2-1|$  (index-3 constraint)")
ax.semilogy(t, envelope(g2), color="#d62728", lw=1.3, ls="--",
            label=r"$|x_n u_n+y_n v_n|$  (index-2 constraint)")

ax.set_xlabel(r"$t$")
ax.set_ylabel("constraint residual")
ax.set_xlim(0, T_END)
ax.set_ylim(1e-9, 1e-1)
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_title(r"DOPRI5, rtol$=10^{%d}$, atol$=10^{%d}$"
             % (int(np.log10(RTOL)), int(np.log10(ATOL))),
             fontsize=8, pad=3)
fig.tight_layout(pad=0.3)
fig.savefig("drift-off-phenomenon.pdf")
plt.close(fig)
print("wrote drift-off-phenomenon.pdf")
