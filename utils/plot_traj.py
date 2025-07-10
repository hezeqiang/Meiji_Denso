import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_frame(x, y, theta, length=1.0, ax=None):
    """
    Plot a 2D frame at (x,y) with rotation theta.
    Red arrow = x‐axis, green arrow = y‐axis.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # unit‐vectors of the frame axes
    ux, uy = np.cos(theta), np.sin(theta)       # x‐axis
    vx, vy = -np.sin(theta), np.cos(theta)      # y‐axis

    # draw arrows
    ax.arrow(x, y, length*ux, length*uy,
             head_width=0.05*length, color='r', linewidth=2)
    ax.arrow(x, y, length*vx, length*vy,
             head_width=0.05*length, color='g', linewidth=2)

    ax.set_aspect('equal')
    ax.grid(True)
    return ax


class FrameAnimator:
    """
    Animate a sequence of 2D frames given poses of shape (N,3):
      poses[i] = [x_i, y_i, theta_i]
    Red arrow = x‐axis, green arrow = y‐axis.
    """

    def __init__(self, poses, length=0.3, interval=200, 
                 xlim=(0,1), ylim=(0,1), grid=True):
        self.poses    = np.asarray(poses)
        assert self.poses.ndim==2 and self.poses.shape[1]==3, \
            "poses must be array of shape (N,3)"
        self.length   = length
        self.interval = interval

        # Set up figure & axes
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        if grid:
            self.ax.grid(True)

        # dummy arrows (will be reset in init/update)
        hw = 0.05 * self.length
        self.x_arrow = self.ax.arrow(0,0, 0,0, head_width=hw, color='r')
        self.y_arrow = self.ax.arrow(0,0, 0,0, head_width=hw, color='g')

    def _init(self):
        # remove and redraw zero‐length arrows
        self.x_arrow.remove()
        self.y_arrow.remove()
        hw = 0.05 * self.length
        self.x_arrow = self.ax.arrow(0,0, 0,0, head_width=hw, color='r')
        self.y_arrow = self.ax.arrow(0,0, 0,0, head_width=hw, color='g')
        return self.x_arrow, self.y_arrow

    def _update(self, i):
        x, y, th = self.poses[i]
        ux, uy = np.cos(th), np.sin(th)
        vx, vy = -np.sin(th), np.cos(th)

        self.x_arrow.remove()
        self.y_arrow.remove()
        hw = 0.05 * self.length

        self.x_arrow = self.ax.arrow(x, y, self.length*ux, self.length*uy,
                                     head_width=hw, color='r')
        self.y_arrow = self.ax.arrow(x, y, self.length*vx, self.length*vy,
                                     head_width=hw, color='g')
        return self.x_arrow, self.y_arrow

    def animate(self):
        """
        Returns a FuncAnimation. Call plt.show() to display.
        """
        return FuncAnimation(self.fig,
                             self._update,
                             frames=len(self.poses),
                             init_func=self._init,
                             interval=self.interval,
                             blit=True)
if __name__ == "__main__":
    # Example usage
    ax = plot_frame(2.0, 1.0, np.pi/4, length=0.8)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    plt.show()


    poses = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 0.1],
        [1.0, 0.6, 0.5],
        # …
    ])

    animator = FrameAnimator(poses, length=0.4, interval=100, xlim=(-1,2), ylim=(-1,2))
    ani = animator.animate()
    plt.show()