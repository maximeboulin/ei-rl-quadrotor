import numpy as np
import pandas as pd
from scipy import signal, linalg
import matplotlib.pyplot as plt
from scipy.signal import correlate
import control
import sys


Ts = 0.05

# Chargement
U = pd.read_csv('bdd_in_mat_05.csv', header=None).values   
Y = pd.read_csv('bdd_out_mat_05.csv', header=None).values
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


# LQR
print("Calcul LQR...")
nk = 1
na, nb = int(sys.argv[1]), int(sys.argv[2])
a, b = arx_multi_run(U_train, Y_train, na, nb, nk)

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
y_sim = simulate_arx(a, b, nk, U_val[r], Y_val[r])

rmse = np.sqrt(np.mean((Y_val[r] - y_sim)**2))
print(f"RMSE sur le run {r} : {rmse:.4f}")


plt.plot(Y_val[r], label='Mesuré')
plt.plot(y_sim, '--', label='Simulé')
plt.legend()
plt.xlabel('Échantillon')
plt.ylabel('Écart à y_eq')
plt.show()