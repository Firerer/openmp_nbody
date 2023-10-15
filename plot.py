import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def read():
    d = []
    for f in glob.glob("log/*.txt"):
        for line in open(f):
            if line.startswith("||"):
                ps = line[2:-2].split(",")
                d.append(
                    [
                        int(ps[0]),
                        int(ps[1]),
                        float(ps[2]),
                        float(ps[3]),
                        float(ps[4]),
                        int(ps[5]),
                    ]
                )
    return np.array(d)


data = read()
sorted_indices = np.lexsort((data[:, -1], data[:, 1], data[:, 0]))
data = data[sorted_indices]
rounds = np.unique(data[:, 1])
max_round = np.max(rounds)
min_round = np.min(rounds)
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 0.8], projection="3d")
ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

s = Slider(
    ax=ax2,
    label="round",
    valmin=min_round,
    valmax=max_round,
    valinit=min_round,
    valstep=rounds,
)

rdata = [data[data[:, 1] == r] for r in rounds]

V = 10


ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")


def update(r):
    ax1.cla()
    ax1.scatter(0, 0, 0, c="k", marker="x")
    ax1.set_xlim([-V, V])  # Set custom X-axis limits
    ax1.set_ylim([-V, V])  # Set custom Y-axis limits
    ax1.set_zlim([-V, V])  # Set custom Z-axis limits
    ax1.autoscale(enable=False, axis="both", tight=True)
    data_r = rdata[int(r)]  # data[(data[:, 1] == r)]
    x = data_r[:, 2]
    y = data_r[:, 3]
    z = data_r[:, 4]
    c = data_r[:, 0]
    ax1.scatter(x, y, z, c=c, marker="o")
    if r > 0:
        last = rdata[int(r) - 1]
        xl = last[:, 2]
        yl = last[:, 3]
        zl = last[:, 4]
        ax1.scatter(xl, yl, zl, marker=".")
        for i in range(len(x)):
            ax1.plot([x[i], xl[i]], [y[i], yl[i]], [z[i], zl[i]])


s.on_changed(update)
update(0)

plt.show()
