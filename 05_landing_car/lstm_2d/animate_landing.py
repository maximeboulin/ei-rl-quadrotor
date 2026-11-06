"""Animation of the landings flown by eval_landing.m on the real drone (uav_plant.m).

Usage:  python 05_landing_car/lstm_2d/animate_landing.py   (from any folder)
Reads eval_landing_results.mat, writes figures/landing_animation.mp4 (and .gif).
Side view, the camera follows the car. One row per flight shown."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyArrow
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_X, TOL_VZ, TOL_VX = 0.10, 0.3, 0.2
ARM = 0.25            # half-width of the drawn drone [m]
FPS = 20              # 1 frame per time step (Ts = 0.05 s) -> real time


def load_flights():
    D = loadmat(os.path.join(HERE, "eval_landing_results.mat"), squeeze_me=True, struct_as_record=False)
    names = list(D["names"])
    speeds = np.atleast_1d(D["car_speeds"])
    seeds = np.atleast_1d(D["seeds"])
    R = D["R"]                                   # (controller, flight)
    flights = {}
    for c, name in enumerate(names):
        i = 0
        for v in speeds:
            for s in seeds:
                L = R[c, i].log
                flights[(name, float(v), int(s))] = {k: np.atleast_1d(getattr(L, k)).astype(float)
                                                     for k in ("t", "x", "z", "vx", "vz", "theta", "x_car", "v_car")}
                flights[(name, float(v), int(s))]["landed"] = bool(L.landed)
                i += 1
    return flights


def verdict(F):
    ex, vz, ev = F["x"][-1] - F["x_car"][-1], F["vz"][-1], F["vx"][-1] - F["v_car"][-1]
    ok = F["landed"] and abs(ex) < TOL_X and abs(vz) < TOL_VZ and abs(ev) < TOL_VX
    if not F["landed"]:
        return "no landing (still in the air)", False
    return ("SUCCESSFUL LANDING" if ok else "touchdown out of tolerance") + \
        f"   x - x_car = {100 * ex:+.0f} cm   vz = {vz:+.2f} m/s   vx - v_car = {ev:+.2f} m/s", ok


def animate(flights, keys, titles, out):
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n))
    axes = np.atleast_1d(axes)
    T = max(len(flights[k]["t"]) for k in keys)
    artists = []
    for ax, k, title in zip(axes, keys, titles):
        F = flights[k]
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-2.3, 1.7)                     # constant width: the box keeps its size when the camera moves
        ax.set_ylim(-0.45, 0.9)
        ax.set_yticks([0, 0.5])
        ax.set_ylabel("z [m]")
        ax.axhline(-0.4, color="0.4", lw=2)                                     # road
        for xm in np.arange(-5, 40, 1.0):                                         # road marks: show the motion
            ax.plot([xm, xm + 0.4], [-0.43, -0.43], color="0.6", lw=3)
        car = Rectangle((0, -0.33), 1.0, 0.28, fc="#3b6ea5", ec="k")             # car body
        plat = Rectangle((0, -0.05), 1.0, 0.05, fc="0.75", ec="k")               # platform (top at z = 0)
        w1, w2 = Circle((0, -0.35), 0.05, fc="k"), Circle((0, -0.35), 0.05, fc="k")
        for p in (car, plat, w1, w2):
            ax.add_patch(p)
        trail, = ax.plot([], [], color="#d95f02", lw=1, alpha=0.6)
        body, = ax.plot([], [], color="k", lw=3, solid_capstyle="round")
        rotors, = ax.plot([], [], "o", color="#d95f02", ms=7)
        target, = ax.plot([], [], "v", color="g", ms=8)
        info = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        end = ax.text(0.5, 0.68, "", transform=ax.transAxes, ha="center", fontsize=10, weight="bold",
                      bbox=dict(fc="white", ec="none", alpha=0.85))
        artists.append(dict(ax=ax, F=F, car=car, plat=plat, w1=w1, w2=w2, trail=trail, body=body,
                            rotors=rotors, target=target, info=info, end=end, verdict=verdict(F)))
    axes[-1].set_xlabel("x [m]  (the camera follows the car)")
    fig.tight_layout()

    def update(f):
        out_artists = []
        for A in artists:
            F = A["F"]
            k = min(f, len(F["t"]) - 1)
            xc, x, z, th = F["x_car"][k], F["x"][k], F["z"][k], F["theta"][k]
            A["ax"].set_xlim(xc - 2.3, xc + 1.7)
            A["car"].set_x(xc - 0.5); A["plat"].set_x(xc - 0.5)
            A["w1"].center = (xc - 0.3, -0.35); A["w2"].center = (xc + 0.3, -0.35)
            # body perpendicular to the thrust (sin th, cos th): pitching forward = accelerating towards +x
            dx, dz = ARM * np.cos(th), -ARM * np.sin(th)
            A["body"].set_data([x - dx, x + dx], [z - dz, z + dz])
            A["rotors"].set_data([x - dx, x + dx], [z - dz + 0.03, z + dz + 0.03])
            A["trail"].set_data(F["x"][:k + 1], F["z"][:k + 1])
            A["target"].set_data([xc], [0.02])
            A["info"].set_text(f"t = {F['t'][k]:5.2f} s   car {F['v_car'][k]:.1f} m/s\n"
                               f"x - x_car = {100 * (x - xc):+5.0f} cm   z = {z:4.2f} m   theta = {np.degrees(th):+5.1f} deg")
            if k == len(F["t"]) - 1:
                text, ok = A["verdict"]
                A["end"].set_text(text); A["end"].set_color("green" if ok else "red")
            else:
                A["end"].set_text("")
            out_artists += [A[n] for n in ("car", "plat", "w1", "w2", "trail", "body", "rotors", "target", "info", "end")]
        return out_artists

    anim = animation.FuncAnimation(fig, update, frames=T + 2 * FPS, interval=1000 / FPS, blit=False)
    anim.save(out + ".mp4", writer=animation.FFMpegWriter(fps=FPS, bitrate=2500), dpi=110)
    anim.save(out + ".gif", writer=animation.PillowWriter(fps=FPS), dpi=60)
    plt.close(fig)
    print("written:", out + ".mp4", out + ".gif")


if __name__ == "__main__":
    flights = load_flights()
    keys = [("SAC", 1.0, 2), ("SAC", 1.5, 1)]
    titles = ["SAC agent (trained on the LSTM) on the real drone, car at 1.0 m/s",
              "SAC agent on the real drone, car at 1.5 m/s"]
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    animate(flights, keys, titles, os.path.join(HERE, "figures", "landing_animation"))
