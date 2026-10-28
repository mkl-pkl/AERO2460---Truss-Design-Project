# =============================================================================
# 3-D "free body" view: the load cloud + hull of every node, drawn on the mount geometry.
# Each node's cloud is positioned so its origin (zero load) sits on that node, i.e. a point
# at node + scale * (Fx, Fy, Fz) is that load case's nodal load vector, applied at the node.
# Based on the original notebook plot (aerospace axes X fwd, Y right, Z down; plotted with
# (x, -y, -z), a proper 180-degree rotation about X, and relabelled Y/Z ticks).
#
# Run after compute_nodal_forces() has been called on every LoadCase in `cases`.
# This file is self-contained (it duplicates the small hull helpers from load_cloud_cells.py).
# -----------------------------------------------------------------------------
# Usage:
# r_nodes = np.array([engine['r_A'], engine['r_B'], engine['r_C'], engine['r_D']])   # [ft]
# plot_free_body_3d(cases, r_nodes).show()
# plot_free_body_3d(cases, r_nodes, include_origin=False, max_len=4).show()
# =============================================================================

import warnings
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
 
NODE_NAMES    = ['A', 'B', 'C', 'D']
GROUP_COLOURS = {'Gust': '#1f77b4', 'Side': '#2ca02c', 'Gyro': '#d62728', 'Manoeuvre': '#9467bd'}   # keys MUST equal the first word of the case names
GROUP_LABELS  = {'Gust': 'Gust load cases', 'Side': 'Side load cases', 'Gyro': 'Gyro load cases', 'Manoeuvre': 'Manoeuvre load cases'}   # legend text (free to edit)
_WARNED = set()
 
 
def _colour(g):
    """Colour for group g (= first word of the case names). Warns once if the group has no entry."""
    if g not in GROUP_COLOURS:
        if g not in _WARNED:
            _WARNED.add(g)
            warnings.warn(f"No colour set for group '{g}': GROUP_COLOURS keys must match the first word of the case "
                          f"names {sorted(GROUP_COLOURS)}. Using black. (Change the legend text via GROUP_LABELS.)",
                          stacklevel=3)
        return 'black'
    return GROUP_COLOURS[g]
 
 
def _label(g):
    return GROUP_LABELS.get(g, g)
_FLIP         = np.array([1.0, -1.0, -1.0])                                # aero frame -> plot frame
 
 
def _cloud_data(cases):
    names  = [c.details['Name'] for c in cases]
    groups = [n.split()[0] for n in names]
    P      = np.array([c.p_nodes for c in cases])           # (n_cases, 4 nodes, 3 components) [lb]
    return names, groups, P
 
 
def _hull(pts, label=None, include_origin=False):
    """ConvexHull of the cases (plus the origin if include_origin), or None if degenerate.
    The origin, when included, is the LAST point: index len(pts). Use hull.points for coordinates.
    If `label` is given, warns when the cases themselves are degenerate (it never fails silently)."""
    pts  = np.asarray(pts, float)
    d    = pts.shape[1]
    rank = np.linalg.matrix_rank(pts - pts.mean(0), tol=1e-9)
    if label and (len(pts) <= d or rank < d):
        warnings.warn(f"{label}: the {len(pts)} load cases span only {rank} of {d} dimensions "
                      f"(coplanar/collinear, or too few distinct cases) - check the inputs.", stacklevel=3)
    X = np.vstack([pts, np.zeros(d)]) if include_origin else pts
    if len(X) <= d or np.linalg.matrix_rank(X - X.mean(0), tol=1e-9) < d:
        return None
    return ConvexHull(X)
 
 
def _hull_edges(hull):
    """Hull edges where the facet orientation changes (hides triangulation diagonals on flat faces)."""
    edges = set()
    for i, simplex in enumerate(hull.simplices):
        for m in range(3):
            j = hull.neighbors[i, m]
            if not np.allclose(hull.equations[i, :-1], hull.equations[j, :-1], atol=1e-6):
                edges.add(tuple(sorted(np.delete(simplex, m))))
    return edges
 
 
def _hover(node, names, loads):
    return [f"Node {node}<br>{n}<br>Fx={p[0]:.1f}  Fy={p[1]:.1f}  Fz={p[2]:.1f}<br>|F|={np.linalg.norm(p):.1f} lb"
            for n, p in zip(names, loads)]
 
 
def _cone_mesh(tips, bases, radius_frac=0.35, m=14):
    """Solid arrowheads: apex at `tips`, circular base centred at `bases` (plot-frame coordinates).
    Built as a plain Mesh3d so the size is exact in scene units (go.Cone sizes depend on point spacing).
    Returns x, y, z, i, j, k lists."""
    X, Y, Z, I, J, K = [], [], [], [], [], []
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    for T, B in zip(tips, bases):
        d  = T - B
        L  = np.linalg.norm(d); d = d / L
        a  = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        ring = B + radius_frac * L * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
        pts  = np.vstack([T, B, ring])                       # 0 = apex, 1 = base centre, 2.. = rim
        o    = len(X)
        X += list(pts[:, 0]); Y += list(pts[:, 1]); Z += list(pts[:, 2])
        for q in range(m):
            q2 = (q + 1) % m
            I += [o, o + 1]; J += [o + 2 + q, o + 2 + q2]; K += [o + 2 + q2, o + 2 + q]   # side + base cap
    return X, Y, Z, I, J, K
 
 
# -----------------------------------------------------------------------------
def plot_free_body_3d(cases, r_nodes, units_per_ft=12, max_len=6, show_hull=True, show_edges=True,
                      include_origin=True, show_box=True, box_depth=1.0, box_opacity=0.35,
                      show_triad=True, lim=None, hull_colour='orange', hull_opacity=0.2,
                      show_arrows=False, arrow_head=None, show_points=True):
    """
    cases          : list of LoadCase (p_nodes already computed)
    r_nodes        : (4, 3) node positions A-D [ft], e.g. np.array([engine[k] for k in ('r_A','r_B','r_C','r_D')])
    units_per_ft   : scene units per ft of node position (12 => scene units are inches, as in the original plot)
    max_len        : scene length of the single largest nodal load vector (one common scale for all nodes)
    include_origin : wrap the node (zero load) into each hull as well
    box_depth      : thickness of the grey mount block behind the node plane [scene units]
    lim            : half-width of the plot cube [scene units]; default = fitted to the data
    hull_colour    : colour of the hull surfaces
    show_arrows    : draw the extreme loads (hull-vertex cases) as arrows from the node to the load point
    arrow_head     : arrowhead length [scene units]; default 0.12 * max_len (shortened on very short arrows)
    show_points    : draw the load-case markers (diamonds / circles). Set False for arrows only; hovering an
                     arrowhead still shows the case name and load components
    """
    names, groups, P = _cloud_data(cases)
    n      = len(cases)
    nodes  = np.asarray(r_nodes, float) * units_per_ft       # (4, 3) scene units, aero frame
    scale  = max_len / np.linalg.norm(P, axis=2).max()       # scene units per lb
    to_pt  = lambda k, loads: (nodes[k] + scale * np.asarray(loads, float)) * _FLIP   # loads [lb] -> plot frame
 
    traces = []
 
    # --- mount block: grey cuboid behind the node plane (as in the original plot) ---------------
    if show_box:
        ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
        zmin, zmax = nodes[:, 2].min(), nodes[:, 2].max()
        Lx, Ly, Lz = box_depth, (ymax - ymin) + 1, (zmax - zmin) + 1
        o  = np.array([nodes[:, 0].min() - Lx, ymin - 0.5, zmin - 0.5])
        ux = np.array([0, 1, 1, 0, 0, 1, 1, 0]); uy = np.array([0, 0, 1, 1, 0, 0, 1, 1]); uz = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        V  = np.column_stack([o[0] + Lx * ux, o[1] + Ly * uy, o[2] + Lz * uz]) * _FLIP
        traces.append(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                                i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 0, 0],
                                j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 3, 7],
                                k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 7, 4],
                                color='grey', opacity=box_opacity, flatshading=True,
                                hoverinfo='skip', showlegend=False))
 
    # --- axis triad at the origin (orientation check, as in the original plot) ------------------
    if show_triad:
        for vec, label, col in [((3, 0, 0), 'X', 'red'), ((0, 3, 0), 'Y', 'green'), ((0, 0, 3), 'Z', 'blue')]:
            tx, ty, tz = np.array(vec) * _FLIP
            traces.append(go.Scatter3d(x=[0, tx], y=[0, ty], z=[0, tz], mode='lines+text',
                                       text=['', label], textposition='top center',
                                       line=dict(color=col, width=6), showlegend=False, hoverinfo='skip'))
 
    # --- load cloud + hull at each node ---------------------------------------------------------
    all_xyz = [V] if show_box else []
    for k in range(4):
        loads = P[:, k, :]                                   # true loads [lb]
        pp    = to_pt(k, loads)                              # plotted positions
        all_xyz.append(pp)
        hull  = _hull(loads, f'Node {NODE_NAMES[k]}', include_origin) if show_hull else None
        crit  = {v for v in hull.vertices if v < n} if hull is not None else set()
 
        if hull is not None:
            hp = to_pt(k, hull.points)                       # hull points (cases [+ node]) in plot frame
            s  = hull.simplices
            traces.append(go.Mesh3d(x=hp[:, 0], y=hp[:, 1], z=hp[:, 2], i=s[:, 0], j=s[:, 1], k=s[:, 2],
                                    color=hull_colour, opacity=hull_opacity, flatshading=True,
                                    hoverinfo='skip', showlegend=False))
            if show_edges:
                ex, ey, ez = [], [], []
                for a, b in _hull_edges(hull):
                    ex += [hp[a, 0], hp[b, 0], None]
                    ey += [hp[a, 1], hp[b, 1], None]
                    ez += [hp[a, 2], hp[b, 2], None]
                traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=2),
                                           hoverinfo='skip', showlegend=False))
 
        if show_arrows and not show_points and k == 0:       # markers normally carry the legend; add entries instead
            for g in dict.fromkeys(groups):
                traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', name=_label(g), legendgroup=g,
                                           line=dict(color=_colour(g), width=6), showlegend=True))
        if show_arrows:                                      # extreme (hull-vertex) loads as arrows, node -> load point
            node_p = nodes[k] * _FLIP
            head   = arrow_head if arrow_head is not None else 0.12 * max_len
            for g in dict.fromkeys(groups):
                ai = [i for i in sorted(crit) if groups[i] == g]
                if not ai:
                    continue
                colour = _colour(g)
                tips   = pp[ai]
                length = np.linalg.norm(tips - node_p, axis=1)
                dirs   = (tips - node_p) / np.maximum(length, 1e-9)[:, None]
                hl     = np.minimum(head, 0.45 * length)                     # head never longer than 45 % of the arrow
                bases  = tips - hl[:, None] * dirs
                sx, sy, sz = [], [], []
                for b in bases:                                              # shafts end where the head starts
                    sx += [node_p[0], b[0], None]; sy += [node_p[1], b[1], None]; sz += [node_p[2], b[2], None]
                traces.append(go.Scatter3d(x=sx, y=sy, z=sz, mode='lines', line=dict(color=colour, width=6),
                                           hoverinfo='skip', legendgroup=g, showlegend=False))
                hx, hy, hz, hi_, hj, hk = _cone_mesh(tips, bases)
                n_vert = len(hx) // len(ai)                                  # vertices per arrowhead
                htxt   = [t for t in _hover(NODE_NAMES[k], [names[i] for i in ai], loads[ai]) for _ in range(n_vert)]
                traces.append(go.Mesh3d(x=hx, y=hy, z=hz, i=hi_, j=hj, k=hk, color=colour, flatshading=True,
                                        hovertext=htxt, hoverinfo='text', legendgroup=g, showlegend=False))
 
        if show_points:
            for g in dict.fromkeys(groups):
                idx = [i for i, gi in enumerate(groups) if gi == g]
                traces.append(go.Scatter3d(
                    x=pp[idx, 0], y=pp[idx, 1], z=pp[idx, 2], mode='markers',
                    marker=dict(size=[7 if i in crit else 4 for i in idx],
                                symbol=['diamond' if i in crit else 'circle' for i in idx],
                                color=_colour(g)),
                    hovertext=_hover(NODE_NAMES[k], [names[i] for i in idx], loads[idx]), hoverinfo='text',
                    name=_label(g), legendgroup=g, showlegend=(k == 0)))
 
    # --- the four nodes (black points, labelled), nudged out along +X so the block doesn't hide them
    eps = 0.1
    nx, ny, nz = ((nodes + np.array([eps, 0, 0])) * _FLIP).T
    traces.append(go.Scatter3d(x=nx, y=ny, z=nz, mode='markers+text', text=NODE_NAMES, textposition='top center',
                               marker=dict(size=6, color='black'), textfont=dict(color='black'),
                               hoverinfo='text', showlegend=False))
 
    # --- layout ---------------------------------------------------------------------------------
    ext = np.abs(np.vstack(all_xyz + [nodes * _FLIP])).max()
    lim = lim or int(np.ceil(ext)) + 2
    ticks   = list(range(-lim, lim + 1, 2))
    flipped = [str(-t) for t in ticks]                       # relabel negated axes
 
    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (fwd)',   range=[-lim, lim], tickvals=ticks),
            yaxis=dict(title='Y (right)', range=[-lim, lim], tickvals=ticks, ticktext=flipped),
            zaxis=dict(title='Z (down)',  range=[-lim, lim], tickvals=ticks, ticktext=flipped),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0), up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=0, b=0))
    key = ('' if not show_points and not show_arrows else
           ' (arrows = extreme / hull-vertex loads)' if not show_points else
           ' (diamond = on hull, circle = interior)')
    fig.add_annotation(text=f'Load scale: 1 scene unit = {1/scale:.0f} lb' + key,
                       xref='paper', yref='paper', x=0.01, y=0.01, showarrow=False, xanchor='left', yanchor='bottom')
    return fig
