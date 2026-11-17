"""PropSim MathKit reference implementations + self-checks.
Every number quoted in MATHKIT.md is printed by this script (run: python mathkit_check.py).
Pure NumPy/SciPy; SI units unless a name says otherwise."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

G0 = 9.80665
PSI = 6894.757293168
LBF = 4.4482216152605
FT = 0.3048
IN = 0.0254
LBM = 0.45359237
E = 1.602176634e-19
EPS0 = 8.8541878128e-12
AMU = 1.66053906660e-27

# ---------------- 1. isentropic nozzle ----------------
def area_ratio(M, g):
    """A/A* from Mach number."""
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M**2)) ** ((g + 1) / (2 * (g - 1)))

def mach_from_area(eps, g, supersonic=True):
    f = lambda M: area_ratio(M, g) - eps
    return brentq(f, 1.0 + 1e-12, 100.0) if supersonic else brentq(f, 1e-9, 1.0 - 1e-12)

def p_ratio(M, g):
    """p/p0."""
    return (1 + 0.5 * (g - 1) * M**2) ** (-g / (g - 1))

def eps_from_pratio(pe_pc, g):
    Me = np.sqrt(2 / (g - 1) * (pe_pc ** (-(g - 1) / g) - 1))
    return area_ratio(Me, g), Me

def Gamma(g):
    """Vandenkerckhove function."""
    return np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))

def CF(g, pe_pc, pa_pc, eps):
    mom = np.sqrt(2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe_pc ** ((g - 1) / g)))
    return mom + (pe_pc - pa_pc) * eps

def cstar(g, RT):
    return np.sqrt(RT) / Gamma(g)

def cstar_from_RT(g, Ru_T_over_M):
    return cstar(g, Ru_T_over_M)

# ---------------- 2. solid motor ----------------
def pc_equilibrium(a, n, rho_p, cs, Ab, At):
    """a in m·s⁻¹·Pa⁻ⁿ: Pc = (a rho_p c* Ab/At)^(1/(1-n))."""
    return (a * rho_p * cs * Ab / At) ** (1 / (1 - n))

def srm_0d_rhs(t, y, a, n, rho_p, cs, At, RT, Ab_of_web):
    """y = [Pc, web, Vc]. V dPc/dt = RT(rho_p - rho_c) Ab r - Pc At RT / c*  (Sutton/Heister 0-D)."""
    Pc, w, V = y
    Ab = Ab_of_web(w)
    r = a * max(Pc, 1.0) ** n
    rho_c = Pc / RT
    dP = (RT * (rho_p - rho_c) * Ab * r - Pc * At * RT / cs) / V
    return [dP, r, Ab * r]

# ---------------- 3. hybrid ----------------
def marxman_rdot(G, x, B, mu, rho_f):
    """rho_f rdot = 0.036 G B^0.23 Re_x^-0.2, Re_x = G x / mu."""
    Re = G * x / mu
    return 0.036 * G * B**0.23 * Re**-0.2 / rho_f

def rdot_radiation(Qc, Qr, rho_f, dHv):
    """rho_f rdot dHv = Qc exp(-Qr/Qc) + Qr (Marxman radiation with blocking)."""
    return (Qc * np.exp(-Qr / Qc) + Qr) / (rho_f * dHv)

def port_radius_analytic(t, R0, a, n, mdot_ox):
    """dR/dt = a (mdot_ox/(pi R^2))^n  ->  R^(2n+1) = R0^(2n+1) + (2n+1) a (mdot_ox/pi)^n t."""
    return (R0 ** (2 * n + 1) + (2 * n + 1) * a * (mdot_ox / np.pi) ** n * t) ** (1 / (2 * n + 1))

def ilfi_tau_bl2(L, Pc, Go, OF, RTav, c=2.05):
    """Karabeyoglu Eq. 59 (Chiaverini Ch. 9): tau = c L Pc / ((2 + 1/(O/F)) Go RT_av)."""
    return c * L * Pc / ((2 + 1 / OF) * Go * RTav)

def ilfi_freq(tau):
    return 0.48 / tau

# Taylor–Culick core flow, nondimensional (radius a, wall injection speed U_w)
def taylor_culick(r, z):
    ur = -np.sin(0.5 * np.pi * r**2) / r
    uz = np.pi * z * np.cos(0.5 * np.pi * r**2)
    return ur, uz

def bidirectional_mantle(kappa, eps):
    """Root of Chiaverini Ch. 6 Eq. 78 (mantle with fuel blowing) in [0.6, 0.97]."""
    def f(b):
        s, c = np.sin(np.pi * b * b), np.cos(np.pi * b * b)
        return c * ((np.pi * kappa + np.pi * eps * (1 + b * b * c)) / s - eps) + np.pi * eps * b * b * s
    return brentq(f, 0.6, 0.97, xtol=1e-14)

def mantle_asymptotic(kappa, eps):
    """Eqs. 79–80: beta* = (A - sqrt C)/B."""
    p = np.pi
    A = p**2 * eps / np.sqrt(2)
    B = 4 * eps - 2 * p * eps + p**2 * eps - 2 * p * kappa
    C = 4 * eps**2 - 6 * p * eps**2 + p**2 * eps**2 - 6 * p * eps * kappa + 4 * p**2 * eps * kappa + 2 * p**2 * kappa**2
    return (A - np.sqrt(C)) / B

# N2O, ESDU 91022 saturation correlations
N2O_TC, N2O_PC, N2O_RHOC = 309.57, 7.251e6, 452.0
def n2o_psat(T):
    t = 1 - T / N2O_TC
    return N2O_PC * np.exp(N2O_TC / T * (-6.71893 * t + 1.35966 * t**1.5 - 1.3779 * t**2.5 - 4.051 * t**5))

def n2o_rho_l(T):
    t = 1 - T / N2O_TC
    return N2O_RHOC * np.exp(1.72328 * t ** (1 / 3) - 0.8395 * t ** (2 / 3) + 0.5106 * t - 0.10412 * t ** (4 / 3))

def n2o_rho_v(T):
    x = N2O_TC / T - 1
    return N2O_RHOC * np.exp(-1.009 * x ** (1 / 3) - 6.28792 * x ** (2 / 3) + 7.50332 * x - 7.90463 * x ** (4 / 3) + 0.629427 * x ** (5 / 3))

# injectors
def mdot_spi(Cd, A, rho, dp):
    return Cd * A * np.sqrt(2 * rho * dp)

def mdot_dyer(Cd, A, rho1, p1, p2, pv, rho2, h1, h2):
    """Dyer NHNE: weighted SPI and HEM, k = sqrt((p1-p2)/(pv-p2))."""
    k = np.sqrt((p1 - p2) / (pv - p2))
    spi = Cd * A * np.sqrt(2 * rho1 * (p1 - p2))
    hem = Cd * A * rho2 * np.sqrt(2 * (h1 - h2))
    return k / (1 + k) * spi + 1 / (1 + k) * hem

# ---------------- 4. liquid engine heat transfer ----------------
def bartz_sigma(Tw_T0, g, M, w=0.6):
    m = 1 + 0.5 * (g - 1) * M**2
    return 1 / ((0.5 * Tw_T0 * m + 0.5) ** (0.8 - w / 5) * m ** (w / 5))

def bartz_h(Dt, mu, cp, Pr, p0, cs, Rc, At_A, sigma):
    """SI Bartz (W·m⁻²·K⁻¹); Rc = throat radius of curvature."""
    return (0.026 / Dt**0.2) * (mu**0.2 * cp / Pr**0.6) * (p0 / cs) ** 0.8 * (Dt / Rc) ** 0.1 * At_A**0.9 * sigma

# ---------------- 5. electric propulsion ----------------
def child_langmuir_j(V, d, M):
    return 4 * EPS0 / 9 * np.sqrt(2 * E / M) * V**1.5 / d**2

def ion_thrust(Ib, Vb, M):
    """Singly charged, no divergence: T = Ib sqrt(2 M Vb / e)."""
    return Ib * np.sqrt(2 * M * Vb / E)

def stuhlinger_payload(dv, ve, vch):
    """m_pl/m0 = exp(-dv/ve) - (ve/vch)^2 (1 - exp(-dv/ve)),  vch^2 = 2 eta t_b / alpha."""
    x = np.exp(-dv / ve)
    return x - (ve / vch) ** 2 * (1 - x)

def edelbaum_dv(v0, v1, di):
    return np.sqrt(v0**2 - 2 * v0 * v1 * np.cos(0.5 * np.pi * di) + v1**2)

# ---------------- 6. level-set burnback (2D, first-order Godunov) ----------------
def levelset_burn(phi, speed, dx, t_end, cfl=0.5):
    """phi < 0 = gas (port), phi > 0 = propellant; front moves along +grad(phi) at speed F = r > 0:
    phi_t + F |grad phi| = 0  (Osher–Sethian). Godunov upwind gradient for F > 0."""
    t = 0.0
    while t < t_end - 1e-15:
        dt = min(cfl * dx / speed, t_end - t)
        p = np.pad(phi, 1, mode='edge')
        dxm = (p[1:-1, 1:-1] - p[1:-1, :-2]) / dx; dxp = (p[1:-1, 2:] - p[1:-1, 1:-1]) / dx
        dym = (p[1:-1, 1:-1] - p[:-2, 1:-1]) / dx; dyp = (p[2:, 1:-1] - p[1:-1, 1:-1]) / dx
        # F > 0 branch: |grad phi|^2 ~ max(D-,0)^2 + min(D+,0)^2 in each direction
        g = np.sqrt(np.maximum(dxm, 0)**2 + np.minimum(dxp, 0)**2 + np.maximum(dym, 0)**2 + np.minimum(dyp, 0)**2)
        phi = phi - dt * speed * g
        t += dt
    return phi

def perimeter(phi, dx, width=1.5):
    """Burning perimeter = integral of delta(phi)|grad phi| (smoothed cosine delta)."""
    gy, gx = np.gradient(phi, dx)
    e = width * dx
    d = np.where(np.abs(phi) < e, (1 + np.cos(np.pi * phi / e)) / (2 * e), 0.0)
    return np.sum(d * np.hypot(gx, gy)) * dx * dx

# ---------------- checks ----------------
def rk4(f, y0, t0, t1, n):
    h = (t1 - t0) / n
    y, t = np.array(y0, float), t0
    for _ in range(n):
        k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1); k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
    return y

if __name__ == '__main__':
    out = []
    def rep(name, val, ref=None, tol=None):
        ok = '' if ref is None else ('PASS' if abs(val - ref) <= tol else 'FAIL')
        out.append(ok)
        print(f'{name:58s} {val:14.6g}' + ('' if ref is None else f'   ref {ref:g} ± {tol:g}  {ok}'))

    print('--- nozzle ---')
    g = 1.2
    eps, Me = eps_from_pratio(1 / 34.0, g)
    rep('eps for g=1.2, Pc/Pe=34 (optimum expansion)', eps)
    rep('Me', Me)
    rep('CF ideal, g=1.2, Pc/Pe=34, pe=pa', CF(g, 1 / 34, 1 / 34, eps))
    rep('A/A* at M=2, g=1.4 (table value 1.6875)', area_ratio(2.0, 1.4), 1.6875, 1e-4)
    rep('M from A/A*=1.6875, g=1.4 (supersonic)', mach_from_area(1.6875, 1.4), 2.0, 1e-4)
    rep('p/p0 at M=2, g=1.4 (table 0.1278)', p_ratio(2.0, 1.4), 0.12780, 5e-5)
    rep('Gamma(1.4) (0.6847)', Gamma(1.4), 0.68473, 5e-5)
    rep('C_F implied by Table 1 LOX/HTPB (Isp 280 s, c* 5972 ft/s) in 1.47-1.56', 280 * G0 / (5972 * FT), 1.515, 0.045)

    print('--- solid ---')
    a_si = 5e-3 * (1e6) ** -0.35  # a such that r = 5 mm/s at 1 MPa, n = 0.35
    Pc = pc_equilibrium(a_si, 0.35, 1750, 1500, 0.5, 2e-3)
    rep('Pc_eq (a: 5 mm/s at 1 MPa, n=.35, rho 1750, c* 1500, Kn 250) MPa', Pc / 1e6)
    rep('  check: mass balance residual (rho a Pc^n Ab - Pc At/c*)/(Pc At/c*)',
        (1750 * a_si * Pc**0.35 * 0.5 - Pc * 2e-3 / 1500) / (Pc * 2e-3 / 1500), 0.0, 1e-9)
    RT = 1500**2 * Gamma(1.2) ** 2
    sol = solve_ivp(srm_0d_rhs, [0, 0.5], [2e5, 0, 0.02], args=(a_si, 0.35, 1750, 1500, 2e-3, RT, lambda w: 0.5),
                    rtol=1e-9, atol=1e-3)
    rep('0-D SRM transient Pc(0.5 s) / Pc_eq (-> ~1 with growing V, rho_c<<rho_p)', sol.y[0, -1] / Pc, 1.0, 0.01)

    print('--- hybrid ---')
    rd = marxman_rdot(200, 0.5, 6, 8e-5, 920)
    rep('Marxman rdot (mm/s): G=200, x=0.5 m, B=6, mu=8e-5, rho_f=920', rd * 1e3)
    rep('radiation-blocked rdot (mm/s): Qc=1 MW/m2, Qr=0.3 MW/m2, 920, 1.81 MJ/kg',
        rdot_radiation(1e6, 3e5, 920, 1.81e6) * 1e3)
    a_h, n_h, R0, mo = 0.0541e-3 * 1.0, 0.647, 0.025, 1.8  # HTPB/GOX: rdot (mm/s) = 0.0541 G^0.647, G kg/m2/s
    f = lambda t, y: np.array([a_h * (mo / (np.pi * y[0] ** 2)) ** n_h])
    Rn = rk4(f, [R0], 0, 10, 2000)[0]
    Ra = port_radius_analytic(10, R0, a_h, n_h, mo)
    rep('port radius after 10 s, analytic (mm)', Ra * 1e3)
    rep('  RK4 vs analytic relative error', abs(Rn - Ra) / Ra, 0.0, 1e-9)
    tau = ilfi_tau_bl2(45 * IN, 568.5 * PSI, 0.48 * LBM / IN**2, 3.01, 6.38e5)
    rep('ILFI NASA Ames 4NF-04 tau_bl2 (ms) [Chiaverini Table 8: 18.2]', tau * 1e3, 18.3, 0.15)
    rep('ILFI frequency 0.48/tau (Hz) [book 26.2, measured 32.75]', ilfi_freq(tau), 26.2, 0.2)
    rep('mantle beta*, Eq. 78, kappa=0.01, eps=0 (inviscid 1/sqrt2=0.707107)', bidirectional_mantle(0.01, 1e-12), 1 / np.sqrt(2), 1e-6)
    rep('mantle beta*, Eq. 78, kappa=0.01, eps=0.001 [book 0.7179]', bidirectional_mantle(0.01, 0.001), 0.7179, 1e-4)
    rep('mantle beta*, Eq. 78, kappa=0.01, eps=0.005 [book 0.7535]', bidirectional_mantle(0.01, 0.005), 0.7535, 1e-4)
    rep('mantle asymptotic Eqs. 79-80, eps=0.001 [book 0.7179]', mantle_asymptotic(0.01, 0.001), 0.7179, 1e-4)
    ur, uz = taylor_culick(1.0, 3.0)
    rep('Taylor-Culick u_r at wall = -1 (unit injection)', ur, -1.0, 1e-12)
    rep('Taylor-Culick u_z at wall = 0 (no slip)', uz, 0.0, 1e-12)
    rep('N2O psat(20 C) MPa (NIST 5.06)', n2o_psat(293.15) / 1e6, 5.06, 0.03)
    rep('N2O rho_l(20 C) (NIST ~786)', n2o_rho_l(293.15), 786, 6)
    rep('N2O rho_v(20 C) (NIST ~160)', n2o_rho_v(293.15), 160, 6)
    rep('N2O psat(Tc) = Pc', n2o_psat(N2O_TC) / 1e6, 7.251, 1e-9)
    rep('SPI mdot: Cd .65, d 1.5 mm, rho 786, dp 2 MPa (kg/s)', mdot_spi(0.65, np.pi * 0.75e-3**2, 786, 2e6))

    print('--- level set ---')
    N = 401; L = 0.2; dx = L / (N - 1); x = np.linspace(-L / 2, L / 2, N); X, Y = np.meshgrid(x, x)
    phi0 = np.hypot(X, Y) - 0.02
    phi1 = levelset_burn(phi0, 0.01, dx, 3.0)       # r = 10 mm/s for 3 s -> R = 50 mm
    Rnum = 0.02 + 0.01 * 3.0
    err = np.max(np.abs(phi1 - (np.hypot(X, Y) - Rnum))[np.abs(np.hypot(X, Y) - Rnum) < 5 * dx])
    rep('level set: circular port 20 -> 50 mm, max |phi - exact| near front (mm)', err * 1e3, 0.0, 2 * dx * 1e3)
    rep('level set: perimeter / (2 pi R) at R = 50 mm', perimeter(phi1, dx) / (2 * np.pi * Rnum), 1.0, 0.01)

    print('--- liquid ---')
    sig = bartz_sigma(0.5, 1.2, 1.0)
    rep('Bartz sigma at throat, Tw/T0=0.5, g=1.2', sig)
    h = bartz_h(0.1, 8e-5, 2000, 0.6, 5e6, 1700, 0.05, 1.0, sig)
    rep('Bartz h (W/m2/K): Dt .1, mu 8e-5, cp 2000, Pr .6, p0 5 MPa, c* 1700', h)

    print('--- electric ---')
    Mxe = 131.293 * AMU
    j = child_langmuir_j(1000, 1e-3, Mxe)
    rep('Child-Langmuir j (A/m2), Xe+, 1 kV, 1 mm', j)
    rep('  scaling check j(2V)/j(V) = 2^1.5', child_langmuir_j(2000, 1e-3, Mxe) / j, 2**1.5, 1e-12)
    T = ion_thrust(1.76, 1100, Mxe)
    rep('ion thrust (mN): Ib=1.76 A, Vb=1100 V, Xe (NSTAR-class ~ 92 mN ideal)', T * 1e3)
    rep('Edelbaum dv LEO(400 km)->GEO, di 28.5 deg (km/s) [~5.9]', edelbaum_dv(7.669e3, 3.075e3, np.radians(28.5)) / 1e3, 5.9, 0.08)
    rep('Stuhlinger payload, dv=5 km/s, ve=vch=20 km/s', stuhlinger_payload(5e3, 2e4, 2e4))

    print('\n', out.count('PASS'), 'PASS', out.count('FAIL'), 'FAIL')
