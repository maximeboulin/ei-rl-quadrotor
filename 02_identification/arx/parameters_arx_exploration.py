import numpy as np
import pandas as pd
from scipy import signal, linalg
import matplotlib.pyplot as plt
from scipy.signal import correlate
import control

Ts = 0.05

# Chargement
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw')
U = pd.read_csv(os.path.join(DATA, 'bdd_in_mat_05.csv'), header=None).values   
Y = pd.read_csv(os.path.join(DATA, 'bdd_out_mat_05.csv'), header=None).values
bad = [122, 455, 1233, 1677]
U, Y = np.delete(U, bad, 0), np.delete(Y, bad, 0)

N_runs, N_steps = U.shape
print(f"Runs: {N_runs}, Steps: {N_steps}, Total: {N_runs*N_steps}")

# Point d'équilibre (hypothèse : chaque run démarre au repos)
u_eq = np.mean(U[:, 0])
y_eq = np.mean(Y[:, 0])

# Écarts
U_d = U - u_eq  
Y_d = Y - y_eq

np.random.seed(13)
indices = np.random.permutation(N_runs)
n_train = int(0.7 * N_runs)

idx_train = indices[:n_train]
idx_val   = indices[n_train:]

U_train = U_d[idx_train, :]  
Y_train = Y_d[idx_train, :]
U_val   = U_d[idx_val, :]
Y_val   = Y_d[idx_val, :]


# estimation du retard de y sur u
nk_max = 20
lags = np.arange(-nk_max, nk_max+1)
C = np.zeros(len(lags))

print("Corrélation croisée en cours...")
for k in range(N_runs):
    c = correlate(U_d[k, :], Y_d[k, :], mode='same', method='direct')
    c = c / np.sqrt(np.sum(U_d[k,:]**2) * np.sum(Y_d[k,:]**2))
    # recentrer sur les lags [-nk_max, nk_max]
    mid = len(c)//2
    C += c[mid-nk_max:mid+nk_max+1]
    if k % 20 == 0 or k == N_runs - 1:
        print(f"  corrélation run {k+1}/{N_runs}")

C /= N_runs
nk_est = np.argmax(np.abs(C)) - nk_max   # négatif si y en avance
nk = max(1, nk_est) if nk_est > 0 else 1

print(f"Retard pur estimé : nk = {nk} échantillons ({nk*Ts:.3f} s)")

# Tracé
plt.subplot(2,1,1)
plt.stem(lags, C)
plt.title('Corrélation croisée moyenne u→y')


# Train ARX
def build_arx_regressors(u_run, y_run, na, nb, nk):
    """
    Pour UN run (vecteurs 1D), construit :
      Phi * theta = Y_target
    avec Phi = [ -y[k-1] ... -y[k-na]  u[k-nk] ... u[k-nk-nb+1] ]
    """
    N = len(u_run)
    rows = N - max(na, nb+nk-1)
    Phi = np.zeros((rows, na + nb))
    Y_target = np.zeros(rows)

    start = max(na, nb+nk-1)
    for i, k in enumerate(range(start, N)):
        # Partie AR
        for j in range(na):
            Phi[i, j] = -y_run[k-1-j]
        # Partie X (entrée)
        for j in range(nb):
            Phi[i, na+j] = u_run[k-nk-j]
        Y_target[i] = y_run[k]
    return Phi, Y_target

def arx_multi_run(U_train, Y_train, na, nb, nk):
    """
    Empile les régresseurs de tous les runs.
    """
    Phi_list, Y_list = [], []
    for r in range(len(U_train)):
        P, Yt = build_arx_regressors(U_train[r], Y_train[r], na, nb, nk)
        Phi_list.append(P)
        Y_list.append(Yt)
    Phi = np.vstack(Phi_list)
    Yvec = np.hstack(Y_list)
    theta, _, _, _ = np.linalg.lstsq(Phi, Yvec, rcond=None)
    a = theta[:na]   # coefficients AR
    b = theta[na:]   # coefficients MA (entrée)
    return a, b

# Balayage des ordres
best_fit = -np.inf
best_order = None
best_coeffs = None

def simulate_arx(a, b, nk, u_input, y_measured):
    """
    Simulation libre (pas one-step-ahead). Initialise avec les premières valeurs mesurées.
    """
    na, nb = len(a), len(b)
    N = len(u_input)
    y_sim = np.zeros(N)
    # initialisation avec les valeurs mesurées
    start = max(na, nb+nk-1)
    y_sim[:start] = y_measured[:start]

    for k in range(start, N):
        # Partie AR
        ar_part = 0.0
        for j in range(na):
            ar_part += -a[j] * y_sim[k-1-j]   # y_sim, pas y_measured (libre !)
        # Partie entrée
        x_part = 0.0
        for j in range(nb):
            x_part += b[j] * u_input[k-nk-j]
        y_sim[k] = ar_part + x_part
    return y_sim


total_combos = 6 * 6 * 3
combo = 0

# na = 1 # tests pour na=1,...,6 mais na=1 suffit
for na in range(1, 7):
    for nb in range(1, 7): # idem que na
        for nk_try in [nk, nk+1, nk+2]:
            a, b = arx_multi_run(U_train, Y_train, na, nb, nk_try)
            # Validation : simulation libre sur chaque run
            fits = []
            for r in range(len(U_val)):
                y_sim = simulate_arx(a, b, nk_try, U_val[r], Y_val[r])
                # FIT % (comme MATLAB)
                nom = np.linalg.norm(Y_val[r] - y_sim)
                den = np.linalg.norm(Y_val[r] - np.mean(Y_val[r]))
                fit = 100 * (1 - nom/(den+1e-10))
                fits.append(fit)
            fit_mean = np.mean(fits)
            if fit_mean > best_fit:
                best_fit = fit_mean
                best_order = (na, nb, nk_try)
                best_coeffs = (a, b)
                print(f"  [{combo}/{total_combos}] na={na} nb={nb} nk={nk_try} "
                      f"FIT={fit_mean:.1f}%  <-- nouveau best")
            else:
                print(f"  [{combo}/{total_combos}] na={na} nb={nb} nk={nk_try} "
                      f"FIT={fit_mean:.1f}%")
            combo += 1

print(f"ARX retenu : na={best_order[0]}, nb={best_order[1]}, nk={best_order[2]}, FIT={best_fit:.1f}%")


# Validation

# LQR
print("Calcul LQR...")
na, nb, nk = best_order
a, b = best_coeffs

# Fonction de transfert : num(z) / den(z) avec retard
# ARX : y + a1 y z^-1 + ... = b1 u z^-nk + b2 u z^{-nk-1} + ...
# =>  den(z) = 1 + a1 z^-1 + ... + a_na z^-na
#     num(z) = b1 z^-nk + ... + b_nb z^{-nk-nb+1}
den = np.hstack([1.0, a])          # 1 + a1 z^-1 + ...
num = np.hstack([np.zeros(nk-1), b])  # zéros pour le retard, puis b
num = np.trim_zeros(num, 'f')      # enlève les zéros inutiles devant

sys_tf = control.TransferFunction(num, den, Ts)
sys_ss = control.tf2ss(sys_tf)     # espace d'état commandable
Ad, Bd, Cd, Dd = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D

print("Pôles :", np.abs(np.linalg.eigvals(Ad)))

Q = np.eye(len(Ad)) * 10
R = np.eye(len(Bd[0,:])) * 0.1
K, _, _ = control.dlqr(Ad, Bd, Q, R)
print("Gain LQR :", K)


# Visualisation
print("Validation finale et tracé...")
r = 0
a, b = best_coeffs
nk = best_order[2]
y_sim = simulate_arx(a, b, nk, U_val[r], Y_val[r])
plt.plot(Y_val[r], label='Mesuré')
plt.plot(y_sim, '--', label='Simulé')
plt.legend()
plt.xlabel('Échantillon')
plt.ylabel('Écart à y_eq')
plt.show()