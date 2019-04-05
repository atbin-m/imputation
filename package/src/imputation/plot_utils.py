import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_diagnostics(t, ytest, ypred, xtest, pred_stats, title):
        
    rmse, r2, mbe = pred_stats
    
    fig = plt.figure(figsize=(15,15))
    fig.suptitle(title, size=15)

    gs = gridspec.GridSpec(4, 3)

    ax2 = plt.subplot(gs[1,:])
    ax1 = plt.subplot(gs[0,:], sharex=ax2, sharey=ax2)
    ax3 = plt.subplot(gs[2,:], sharex=ax2, sharey=ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax41 = plt.subplot(gs[3,0]) 
    ax42 = plt.subplot(gs[3,1]) 
    ax43 = plt.subplot(gs[3,2]) 

    # Plotting top 3 rows
    ax1.plot(t, ytest, lw=2., color='C0', label='Observed')
    ax1.set_xlim(t.min(), t.max())
    ax1.legend()

    for col in xtest.iteritems():
        ax2.plot(t, col[1], label = col[0])
    ax2.set_xlim(t.min(), t.max())
    ax2.legend(ncol=3)

    ax3.plot(t, ytest, lw=2., color='C0', label='Observed')
    ax3.plot(t, ypred, lw=2., color='C3', ls='solid', alpha=0.6, label='Predicted')
    ax3.legend(ncol=2)

    # Plotting bottom row (2 columns)
    density_plot = ax41.hexbin(ytest, ypred, bins='log', mincnt=2, gridsize=50)
    ax41.plot([ytest.min(), ytest.max()],[ytest.min(), ytest.max()], 'k', lw=2.)
    ax41.set_xlabel('Observed')
    ax41.set_ylabel('Predicted')
    ax41.set_aspect(0.9)

    # Printing stats in the axes
    ax42.text(0.5, 0.8, 'No. points: %d'%ytest.shape[0], fontsize=18)
    #ax42.text(0.5, 0.7, 'No. filled: %d'%ytest[ytest.isna()].shape[0], fontsize=18)
    ax42.text(0.5, 0.6, 'RMSE: %1.3f'%rmse, fontsize=18)
    ax42.text(0.5, 0.5, 'R2: %1.3f'%r2, fontsize=18)
    ax42.text(0.5, 0.4, 'MBE: %1.3f'%mbe, fontsize=18)
    ax42.text(0.5, 0.3, 'Var(Tower): %1.3f'%np.var(ytest), fontsize=18)
    ax42.text(0.5, 0.2, 'Var(Predicted): %1.3f'%np.var(ypred), fontsize=18)

    ax42.axis('off')
    ax43.axis('off')

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.minorticks_on()
    
    return fig, gs


#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) implementation.
Note: If you have found these software useful for your research, I would
appreciate an acknowledgment.

https://gist.github.com/ycopin/3342888
"""

__version__ = "Time-stamp: <2018-12-06 11:43:41 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours
    

def taylor_diagram(samples, refstd, srange, title):
    """
    samples = [ Solvers X 3]
    """
    fig = plt.figure(figsize=(5,5))
    fig.suptitle(title, size=12)

    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, label="Reference", srange=srange)

    colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))

    # Add the models to Taylor diagram
    for i, (stddev, corrcoef, asolver_name) in enumerate(samples):
        dia.add_sample(float(stddev), float(corrcoef),
                       marker='$%d$' % (i+1), ms=10, ls='',
                       mfc=colors[i], mec=colors[i],
                       label=asolver_name)

    # Add grid
    dia.add_grid()

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    # Add a figure legend
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')
    
    return fig