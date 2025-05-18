from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.io import to_image
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from elastory.network.network import Network
from elastory.utils.f_calculate_rmsd import fitAtoB


def add_interior_points(fig, points, hull_mask):
    fig.add_trace(
        go.Scatter3d(
            x=points[~hull_mask, 0],
            y=points[~hull_mask, 1],
            z=points[~hull_mask, 2],
            mode="markers",
            marker=dict(color="cyan", size=5),
            name="Interior Points",
        )
    )
    return fig


def add_hull_points(fig, points, hull_mask):
    fig.add_trace(
        go.Scatter3d(
            x=points[hull_mask, 0],
            y=points[hull_mask, 1],
            z=points[hull_mask, 2],
            mode="markers",
            marker=dict(color="orange", size=5),
            name="Hull Points",
        )
    )
    return fig


def add_hull_lines(fig, points, hull):
    for simplex in hull.simplices:
        fig.add_trace(
            go.Scatter3d(
                x=points[simplex, 0],
                y=points[simplex, 1],
                z=points[simplex, 2],
                mode="lines",
                line=dict(color="grey", width=2),
                showlegend=False,
            )
        )
    return fig


def add_hull_surfaces(fig, points, hull):
    for simplex in hull.simplices:
        fig.add_trace(
            go.Mesh3d(
                x=points[simplex, 0],
                y=points[simplex, 1],
                z=points[simplex, 2],
                opacity=0.2,
                color="rgba(100,100,100,0.5)",
                hoverinfo="none",
                showlegend=False,
            )
        )
    return fig


def add_node_trace(fig, pos_arr, node_sizes, node_colors, node_alpha, labels):
    fig.add_trace(
        go.Scatter3d(
            x=pos_arr[:, 0],
            y=pos_arr[:, 1],
            z=pos_arr[:, 2],
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=node_alpha,
                line=dict(width=int(1e8), color="rgb(1,1,1)"),
            ),
            text=[str(i) for i in range(len(pos_arr))],
            hoverinfo="text" if labels else "none",
            name="nodes",
        )
    )
    return fig


def add_edge_trace(fig, G, pos_arr, lw=1):
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos_arr[edge[0]]
        x1, y1, z1 = pos_arr[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=lw, color="#888"),
            hoverinfo="none",
            mode="lines",
            name="edges",
        )
    )
    return fig


def add_pulling_traces(fig, net):
    if net.source_pairs is not None:
        for pair in net.source_pairs:
            fig.add_trace(
                go.Scatter3d(
                    x=net.geometry.bead_positions[list(pair), 0],
                    y=net.geometry.bead_positions[list(pair), 1],
                    z=net.geometry.bead_positions[list(pair), 2],
                    mode="lines",
                    line=dict(color="darkblue", width=10),
                    opacity=0.7,
                    hoverinfo="none",
                    name="Pulling Connections",
                )
            )
    return fig


def update_layout(
    fig,
    theme=None,
    hide_axes=False,
    show_legend=True,
    width=750.0,
    height=750.0,
    zoom=1.0,
):
    layout_update = {
        "scene": {
            "xaxis": {"visible": not hide_axes, "title": "X"},
            "yaxis": {"visible": not hide_axes, "title": "Y"},
            "zaxis": {"visible": not hide_axes, "title": "Z"},
            "aspectmode": "data",
        },
        "showlegend": show_legend,
        "width": width,
        "height": height,
        "margin": dict(t=25, r=0, l=25, b=25),
        "legend": dict(x=0.8, y=0.9),
    }

    if theme:
        layout_update["template"] = theme

    layout_update["scene"]["camera"] = dict(eye=dict(x=zoom * 1.25, y=zoom * 1.25, z=zoom * 0.1))

    fig.update_layout(**layout_update)
    return fig


def update_node_colors(
    net: Network,
    positions: np.ndarray,
    repulsive_cutoff: float,
    node_colors: list,
) -> list:
    """
    Update node colors based on touching condition.

    Args:
        positions (np.ndarray): Array of node positions.
        repulsive_cutoff (float): Cutoff distance.
        node_colors (list): List of current node colors.

    Returns:
        list: Updated list of node colors.
    """
    pdr = pdist(positions)
    conn = np.nonzero(pdr < 2 * repulsive_cutoff)[0]
    if conn.shape[0] > 0:
        conn = net.dense_square_map(conn)
        conn_inds = np.unique(conn)
        for p in conn_inds:
            node_colors[p] = "orange"
    return node_colors


def create_animation_buttons(speed: int) -> dict:
    """
    Create animation buttons for the figure.

    Args:
        speed (int): speed factor for animation speed.

    Returns:
        dict: Dictionary containing animation button settings.
    """
    return dict(
        type="buttons",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": int(50 / speed), "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            ),
        ],
    )


def plot_3d_convex_hull(
    fig: go.Figure,
    points: np.ndarray,
) -> go.Figure:
    hull = ConvexHull(points)
    hull_mask = np.zeros(len(points), dtype=bool)
    hull_mask[hull.vertices] = True

    fig = add_interior_points(fig, points, hull_mask)
    fig = add_hull_points(fig, points, hull_mask)
    fig = add_hull_lines(fig, points, hull)
    fig = add_hull_surfaces(fig, points, hull)

    return fig


def plot_network(
    fig: go.Figure,
    net: Network,
    labels=True,
    pulling_connections=False,
    connections=True,
    edge_width=1,
    node_colors=None,
    edge_colors=None,
):
    G = net.graph.graph

    # Node properties
    node_colors = [G.nodes[u]["color"] for u in G.nodes()]
    node_sizes = [G.nodes[u]["size"] for u in G.nodes()]
    node_alpha = 0.7

    # Adjust node sizes
    max_size = max(node_sizes)
    node_sizes = [5 + (n / max_size) * 15 for n in node_sizes]
    if labels:
        node_sizes = [n * 1.5 for n in node_sizes]

    pos_arr = net.geometry.bead_positions

    fig = add_node_trace(fig, pos_arr, node_sizes, node_colors, node_alpha, labels)

    if connections:
        fig = add_edge_trace(fig, G, pos_arr, lw=edge_width)

    if pulling_connections:
        fig = add_pulling_traces(fig, net)

    return fig


def response_video(
    fig: go.Figure,
    net: Network,
    trajectory: np.ndarray,
    skip: int = 1,
    new_connections: bool = False,
    touching: bool = True,
    speed: int = 2,
    lw: int = 1,
) -> go.Figure:
    """
    Create an animated video of network response.

    Args:
        fig (go.Figure): Plotly figure object to add traces to.
        net (Network): Network object containing graph and geometry information.
        trajectory (np.ndarray): Array of response positions over time.
        skip (int): Number of frames to skip between each plotted frame.
        touching (bool): Whether to highlight touching nodes.
        speed (int): speed factor for animation speed.
        lw (int): Line width for edges.

    Returns:
        go.Figure: Updated Plotly figure with animation.
    """

    poses_to_plot = trajectory[::skip]
    num_preloads = poses_to_plot.shape[0]
    pos_fit_to = poses_to_plot[0]

    G = net.graph.graph
    pos_arr = net.geometry.bead_positions

    # Node properties
    node_colors = [G.nodes[u]["color"] for u in G.nodes()]
    node_sizes = [G.nodes[u]["size"] for u in G.nodes()]
    node_alpha = 0.7

    # Adjust node sizes
    max_size = max(node_sizes)
    node_sizes = [5 + (n / max_size) * 15 for n in node_sizes]

    # Add initial traces with labels
    fig = add_node_trace(fig, pos_arr, node_sizes, node_colors, node_alpha, labels=True)
    fig = add_edge_trace(fig, G, pos_arr, lw=lw)

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # Create frames for animation
    frames = []
    for i in range(num_preloads):
        tmp_pos = fitAtoB(poses_to_plot[i], pos_fit_to)

        if touching and net.geometry.repulsive_cutoff is not None:
            node_colors = update_node_colors(
                net, poses_to_plot[i], net.geometry.cutoff_length, node_colors
            )

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=tmp_pos[:, 0],
                    y=tmp_pos[:, 1],
                    z=tmp_pos[:, 2],
                    mode="markers",  # Remove 'text' mode to hide labels
                    marker=dict(size=node_sizes, color=node_colors, opacity=node_alpha),
                    hoverinfo="none",
                ),
                *add_edge_trace(go.Figure(), G, tmp_pos, lw=lw).data,
                *add_pulling_traces(go.Figure(), net).data,
            ]
        )
        frames.append(frame)

    fig.frames = frames

    # Add animation buttons
    fig.update_layout(updatemenus=[create_animation_buttons(speed)])

    return fig


def add_trace(
    fig: go.Figure,
    positions: np.ndarray,
    name: str | None,
    color: str = "blue",
    alpha: float = 1.0,
    size=1,
    legendrank: int | None = None,
):
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                opacity=alpha,
            ),
            name=name if name else None,
            legendrank=legendrank if legendrank else None,
        )
    )


def plot_network_motion(
    fig: go.Figure,
    net: Network,
    trajectory: np.ndarray,
    plot_source=True,
    plot_target=True,
    size_factor=1,
    alpha: float = 1.0,
    name: str = "Response Positions",
):
    # Add response positions
    trajectory_flat = trajectory.reshape(-1, 3)
    # make color gradient for steps
    # num_steps = trajectory.shape[0]
    # num_nodes = trajectory.shape[1]
    # steps_color = np.repeat(np.arange(num_steps), num_nodes)

    add_trace(
        fig=fig,
        positions=trajectory_flat,
        name=name,
        color="darkgreen",
        size=1 * size_factor,
        alpha=alpha,
        legendrank=1,
    )

    if plot_source:
        # Add source positions
        source_positions = trajectory[:, net.source, :].reshape(-1, 3)
        add_trace(
            fig=fig,
            positions=source_positions,
            name="source",
            color="blue",
            size=size_factor,
            alpha=alpha,
        )

    if plot_target:
        # Add target positions
        target_positions = trajectory[:, net.target, :].reshape(-1, 3)
        add_trace(
            fig=fig,
            positions=target_positions,
            name="target",
            color="red",
            size=size_factor,
            alpha=alpha,
        )

    return fig


def net_to_img(plotly_fig: go.Figure) -> np.ndarray:
    """
    Convert a Plotly figure to a numpy array with an alpha channel.
    This is useful for combining Plotly figures with other plotting libraries.
    E.g. for matplotlib, use:
    ```
    ax.imshow(img_array)
    ```
    """

    # Convert Plotly figure to image
    img_bytes = to_image(plotly_fig, format="png", scale=2)

    # Convert bytes to numpy array with alpha channel
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")
    img_array = np.array(img)
    return img_array
