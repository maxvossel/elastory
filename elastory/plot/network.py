import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colormaps
from matplotlib.colors import BoundaryNorm, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import pdist, squareform

from ..utils.f_calculate_rmsd import fitAtoB

####################
# 3D network plots #
####################


def netplot_3D(
    snet,
    ax,
    angle1=70,
    angle2=250,
    hide_axes=True,
    labels=False,
    pulling_connections=False,
    connections=True,
    fontsize=15,
    scale_fac=1,
    lw=1.1,
):
    G = snet.graph

    # Get node positions
    # pos = nx.get_node_attributes(G, 'pos') @TODO remove this line?
    pos = dict(zip(range(snet.N), snet.bead_positions))

    # Get number of nodes
    n = G.number_of_nodes()

    # Define color range proportional to number of edges adjacent
    # to a single node
    node_colors = [G.nodes[(u)]["color"] for u in G.nodes()]
    node_sizes = [G.nodes[(u)]["size"] for u in G.nodes()]
    node_alpha = 0.7
    if labels:
        node_sizes = [n * 3 for n in node_sizes]
        node_alpha = 0.3

    # 3D network plot
    with plt.style.context(("ggplot")):
        pos_arr = snet.bead_positions

        points = ax.scatter(
            *pos_arr.T, c=node_colors, s=node_sizes, edgecolors="k", alpha=node_alpha
        )

        if labels:
            for n, label in enumerate(G.nodes):
                ax.text(*pos_arr[n] - 0.45, label, fontsize=fontsize)

        if connections:
            # Loop on the list of edges to get the x,y,z, coordinates of the
            # connected nodes
            # Those two points are the extrema of the line to be plotted

            x, y, z = np.zeros((3, 3 * G.number_of_edges()))
            L = [x, y, z]

            edges = np.array([*G.edges()])
            values = np.array([*pos.values()], dtype=float)
            keys = np.array([*pos.keys()], dtype=int)

            for i in (0, 1, 2):
                pos = dict(zip(keys, values[:, i]))
                L[i][0::3] = np.vectorize(pos.get)(edges[:, 0])
                L[i][1::3] = np.vectorize(pos.get)(edges[:, 1])
                L[i][2::3] = None

            # Plot the connecting lines
            (lines,) = ax.plot(x, y, z, c="black", alpha=0.5, lw=lw)

    if pulling_connections:
        for pair in snet.source_pairs:
            ax.plot(*snet.bead_positions[[*pair]].T, c="darkblue", lw=1, alpha=0.7, ls="-")

    # Set the initial view
    ax.view_init(angle1, angle2)

    # Hide the axes
    if hide_axes:
        ax.set_axis_off()
    if connections:
        return points, lines
    else:
        return points


def response_video(
    snet,
    response_pos,
    fig,
    ax,
    skip=1,
    new_connections=False,
    touching=True,
    angle1=70,
    angle2=260,
    scale=2,
    lw=1,
):
    poses_to_plot = response_pos[::skip]
    num_preloads = poses_to_plot.shape[0]

    points, lines = netplot_3D(snet, ax, angle1, angle2, hide_axes=True, lw=lw)
    colors = points.get_facecolors()

    ax.set_xlim(tuple(np.array(ax.get_xlim()) * 0.6))
    ax.set_ylim(tuple(np.array(ax.get_ylim()) * 0.6))
    ax.set_zlim(tuple(np.array(ax.get_zlim()) * 0.6))

    plt.close()

    dist_mat = np.triu(squareform(pdist(snet.bead_positions)))
    edges_init = np.array(np.where((dist_mat > 0) & (dist_mat < snet.cutoff_length))).T

    pos_fit_to = poses_to_plot[0]

    def animate(i):
        print(i, " of ", num_preloads - 1, end="\r")
        if i < num_preloads:
            j = i
        else:
            j = num_preloads - 1 - i

        pos_to_fit = poses_to_plot[j]
        tmp_pos = fitAtoB(pos_to_fit, pos_fit_to)
        points._offsets3d = tmp_pos.T

        keys = np.arange(len(tmp_pos))
        values = tmp_pos

        if new_connections:
            dist_mat = np.triu(squareform(pdist(tmp_pos)))
            edges = np.array(np.where((dist_mat > 0) & (dist_mat < snet.cutoff_length))).T
        else:
            edges = edges_init

        lx, ly, lz = np.zeros((3, 3 * len(edges)))
        L = [lx, ly, lz]

        for i in (0, 1, 2):
            pos = dict(zip(keys, values[:, i]))
            L[i][0::3] = np.vectorize(pos.get)(edges[:, 0])
            L[i][1::3] = np.vectorize(pos.get)(edges[:, 1])
            L[i][2::3] = None

        lines.set_data(lx, ly)
        lines.set_3d_properties(lz)

        if touching:
            pdr = pdist(poses_to_plot[j])
            conn = np.nonzero(pdr < 2 * snet.repulsive_cutoff)[0]
            if conn.shape[0] > 0:
                conn = snet.dense_square_map(conn)
                conn_inds = np.unique(conn)
                for p in conn_inds:
                    colors[p] = np.array(to_rgba("orange", alpha=0.7))

        return points, lines

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_preloads,
        interval=50 * scale,
        repeat=True,
        blit=True,
    )
    return anim


def strain_video(snet, optimizer, fig, ax, skip=1):
    response_pos = snet.response_data[optimizer]["pos"]
    poses_to_plot = response_pos[:-1:skip]
    num_preloads = poses_to_plot.shape[0]
    pos = response_pos[0]

    points = ax.scatter(*pos.T, c="black", s=10, edgecolors="k")

    ax.set_xlim(tuple(np.array(ax.get_xlim()) * 0.6))
    ax.set_ylim(tuple(np.array(ax.get_ylim()) * 0.6))
    ax.set_zlim(tuple(np.array(ax.get_zlim()) * 0.6))

    plt.close()

    d0mats = np.array([squareform(pdist(pos)) for pos in response_pos])
    d0mat_diffs = np.diff(d0mats, axis=0)

    cmap = colormaps.get_cmap("RdBu")
    norm = BoundaryNorm(np.linspace(-0.01, +0.01, cmap.N), cmap.N)

    segments = pos[np.array([*snet.graph.edges()])]
    segment_colors = np.array([d0mat_diffs[0][tup] for tup in [*snet.graph.edges()]])

    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(segment_colors)
    lc.set_linewidth(np.abs(segment_colors) / np.max(np.abs(segment_colors)) * 15)

    lines = ax.add_collection(lc)
    pos_fit_to = response_pos[0]

    def animate(i):
        print(i + 1, " of ", num_preloads, end="\r")
        if i < num_preloads:
            j = i
        else:
            j = num_preloads - 1 - i

        pos_to_fit = poses_to_plot[j]
        tmp_pos = fitAtoB(pos_to_fit, pos_fit_to)

        segments = tmp_pos[np.array([*snet.graph.edges()])]
        segment_colors = np.array([d0mat_diffs[i][tup] for tup in [*snet.graph.edges()]])

        lines.set_segments(segments)
        lines.set_cmap(cmap)
        lines.set_norm(norm)
        lines.set_array(segment_colors)
        lines.set_linewidth(0.5 + np.abs(segment_colors) / np.max(np.abs(segment_colors)) * 15)

        x, y, z = tmp_pos.T
        points._offsets3d = x, y, z

        lines.set_zorder(0.1)
        points.set_zorder(1)

        return lines, points

    anim = animation.FuncAnimation(
        fig, animate, frames=(num_preloads), interval=50, repeat=True, blit=True
    )

    return anim
