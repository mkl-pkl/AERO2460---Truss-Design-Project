# =============================================================================
# LOAD CLOUD: nodal force vectors for every load case, with convex hull
# Run after compute_nodal_forces() has been called on every LoadCase in `cases`.
# Axes: X forward, Y right, Z down. The 3-D view plots are oriented (x, -y, -z)
# (a proper 180-degree rotation about X, so "down" is down on screen) and relabels the
# Y and Z ticks with the true values.
# -----------------------------------------------------------------------------
# Usage:
# plot_load_cloud_3d(cases).show()
# plot_load_cloud_2d(cases).show()
# =============================================================================

import warnings
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
 
NODE_NAMES    = ['A', 'B', 'C', 'D']
GROUP_COLOURS = {'Gust': '#1f77b4', 'Side': '#2ca02c', 'Gyro': '#d62728'}   # keys MUST equal the first word of the case names
GROUP_LABELS  = {'Gust': 'Gust load cases', 'Side': 'Side load cases', 'Gyro': 'Gyro load cases'}   # legend text (free to edit)
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
 
 
def _cloud_data(cases):
    names  = [c.details['Name'] for c in cases]
    groups = [n.split()[0] for n in names]
    P      = np.array([c.p_nodes for c in cases])           # (n_cases, 4 nodes, 3 components)
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
    """3-D hull edges where the facet orientation changes (hides triangulation diagonals on flat faces)."""
    edges = set()
    for i, simplex in enumerate(hull.simplices):
        for m in range(3):
            j = hull.neighbors[i, m]                        # facet sharing the edge opposite vertex m
            if not np.allclose(hull.equations[i, :-1], hull.equations[j, :-1], atol=1e-6):
                edges.add(tuple(sorted(np.delete(simplex, m))))
    return edges
 
 
def _hover(names, pts):
    return [f"{n}<br>Fx={p[0]:.1f}  Fy={p[1]:.1f}  Fz={p[2]:.1f}<br>|F|={np.linalg.norm(p):.1f} lb"
            for n, p in zip(names, pts)]
 
 
def _padded_limits(P, frac=0.08):
    flat   = np.vstack([P.reshape(-1, 3), np.zeros(3)])      # always include the origin
    lo, hi = flat.min(0), flat.max(0)
    pad    = frac * np.maximum(hi - lo, 1e-9)
    return lo - pad, hi + pad                                # shared by all nodes => panels directly comparable
 
 
def _cube_limits(lo, hi):
    """Equal span on all three axes (centred on each axis's data), so a 'cube' scene is true to scale."""
    centre = (lo + hi) / 2
    half   = (hi - lo).max() / 2
    return centre - half, centre + half
 
 
def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw  = span / n
    mag  = 10 ** np.floor(np.log10(raw))
    step = mag * min((1, 2, 5, 10), key=lambda m: abs(m * mag - raw))
    return np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
 
 
# -----------------------------------------------------------------------------
def plot_load_cloud_3d(cases, show_hull=True, show_edges=True, include_origin=True, rays=None):
    """One 3-D panel per node. Each load case is a point (Fx, Fy, Fz) [lb]; black x = origin (zero load).
    Grey solid = convex hull (wrapped around the origin too if include_origin=True);
    diamonds = cases on the hull (critical for that node); circles = interior cases.
    show_edges=False hides the black hull edge lines (translucent hull surface only).
    rays='all' or 'hull' adds thin lines from the origin to the cases (off by default)."""
    names, groups, P = _cloud_data(cases)
    n    = len(cases)
    lo, hi = _cube_limits(*_padded_limits(P))
    flip = np.array([1, -1, -1])                             # plot frame = (x, -y, -z)
 
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'scene'}] * 2] * 2,
                        subplot_titles=[f'Node {n_}' for n_ in NODE_NAMES],
                        vertical_spacing=0.04, horizontal_spacing=0.02)
 
    for k in range(4):
        row, col = k // 2 + 1, k % 2 + 1
        pts  = P[:, k, :]                                    # true values (used for hull + hover)
        pp   = pts * flip                                    # plotted positions
        hull = _hull(pts, f'Node {NODE_NAMES[k]}', include_origin) if show_hull else None
        crit = {v for v in hull.vertices if v < n} if hull is not None else set()   # cases only, not the origin
 
        if hull is not None:
            hp = hull.points * flip                          # hull coordinates (cases [+ origin]) in plot frame
            s  = hull.simplices
            fig.add_trace(go.Mesh3d(x=hp[:, 0], y=hp[:, 1], z=hp[:, 2], i=s[:, 0], j=s[:, 1], k=s[:, 2],
                                    color='grey', opacity=0.15, flatshading=True,
                                    hoverinfo='skip', showlegend=False), row=row, col=col)
            if show_edges:
                ex, ey, ez = [], [], []
                for a, b in _hull_edges(hull):
                    ex += [hp[a, 0], hp[b, 0], None]
                    ey += [hp[a, 1], hp[b, 1], None]
                    ez += [hp[a, 2], hp[b, 2], None]
                fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=2),
                                           hoverinfo='skip', showlegend=False), row=row, col=col)
 
        # origin (zero load)
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                                   marker=dict(size=3, symbol='x', color='black', line=dict(width=1)),
                                   hovertext='origin (zero load)', hoverinfo='text',
                                   name='origin (zero load)', legendgroup='origin', showlegend=(k == 0)),
                      row=row, col=col)
 
        for g in dict.fromkeys(groups):                      # unique groups, in order of appearance
            colour = _colour(g)
            idx    = [i for i, gi in enumerate(groups) if gi == g]
 
            ridx = idx if rays == 'all' else [i for i in idx if i in crit] if rays == 'hull' else []
            if ridx:                                         # rays: origin -> each case
                rx, ry, rz = [], [], []
                for i in ridx:
                    rx += [0, pp[i, 0], None]; ry += [0, pp[i, 1], None]; rz += [0, pp[i, 2], None]
                fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines', opacity=0.5,
                                           line=dict(color=colour, width=2), hoverinfo='skip',
                                           legendgroup=g, showlegend=False), row=row, col=col)
 
            fig.add_trace(go.Scatter3d(
                x=pp[idx, 0], y=pp[idx, 1], z=pp[idx, 2], mode='markers',
                marker=dict(size=[7 if i in crit else 4 for i in idx],
                            symbol=['diamond' if i in crit else 'circle' for i in idx], color=colour),
                hovertext=_hover([names[i] for i in idx], pts[idx]), hoverinfo='text',
                name=_label(g), legendgroup=g, showlegend=(k == 0)), row=row, col=col)
 
    tx, ty, tz = (_nice_ticks(lo[i], hi[i]) for i in range(3))
    fig.update_scenes(
        xaxis=dict(title='Fx (fwd) [lb]',   range=[lo[0], hi[0]],   tickvals=tx,  ticktext=[f'{t:g}' for t in tx]),
        yaxis=dict(title='Fy (right) [lb]', range=[-hi[1], -lo[1]], tickvals=-ty, ticktext=[f'{t:g}' for t in ty]),
        zaxis=dict(title='Fz (down) [lb]',  range=[-hi[2], -lo[2]], tickvals=-tz, ticktext=[f'{t:g}' for t in tz]),
        aspectmode='cube',                                   # equal spans on all axes => true scale
        camera=dict(eye=dict(x=1.9, y=1.9, z=1.2), up=dict(x=0, y=0, z=1)))
    fig.update_layout(height=850, margin=dict(l=0, r=0, t=60, b=0),
                      title='Nodal load cloud per node (diamond = on convex hull, circle = interior)')
    return fig
 
 
# -----------------------------------------------------------------------------
def plot_load_cloud_2d(cases, show_hull=True, include_origin=True, rays=None):
    """Rows = nodes, columns = projections (top, side, rear). Shaded polygon = 2-D hull of the projection.
    Diamonds = cases on the *3-D* hull of that node, so the markers match the 3-D figure.
    rays='all' or 'hull' adds thin lines from the origin (x) to the cases (off by default)."""
    names, groups, P = _cloud_data(cases)
    n    = len(cases)
    lo, hi = _padded_limits(P)
 
    # (horizontal component, vertical component, title, reverse vertical axis?)
    planes = [(0, 1, 'Fx-Fy (top view)',  True),
              (0, 2, 'Fx-Fz (side view)', True),
              (1, 2, 'Fy-Fz (rear view)', True)]
    label  = ['Fx (fwd) [lb]', 'Fy (right) [lb]', 'Fz (down) [lb]']
 
    fig = make_subplots(rows=4, cols=3, column_titles=[p[2] for p in planes],
                        row_titles=[f'Node {n_}' for n_ in NODE_NAMES],
                        horizontal_spacing=0.07, vertical_spacing=0.05)
 
    for k in range(4):
        pts3 = P[:, k, :]
        h3   = _hull(pts3, f'Node {NODE_NAMES[k]}', include_origin) if show_hull else None
        crit = {v for v in h3.vertices if v < n} if h3 is not None else set()
 
        for c, (ih, iv, _, rev) in enumerate(planes, start=1):
            pts2 = pts3[:, [ih, iv]]
            h2   = _hull(pts2, include_origin=include_origin) if show_hull else None
            if h2 is not None:
                X2 = h2.points
                v  = list(h2.vertices) + [h2.vertices[0]]                  # CCW order, closed
                fig.add_trace(go.Scatter(x=X2[v, 0], y=X2[v, 1], mode='lines', fill='toself',
                                         fillcolor='rgba(128,128,128,0.15)', line=dict(color='black', width=1),
                                         hoverinfo='skip', showlegend=False), row=k + 1, col=c)
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=7, symbol='x', color='black'),
                                     hovertext='origin (zero load)', hoverinfo='text', name='origin (zero load)',
                                     legendgroup='origin', showlegend=(k == 0 and c == 1)), row=k + 1, col=c)
            for g in dict.fromkeys(groups):
                colour = _colour(g)
                idx    = [i for i, gi in enumerate(groups) if gi == g]
                ridx   = idx if rays == 'all' else [i for i in idx if i in crit] if rays == 'hull' else []
                if ridx:
                    rx, ry = [], []
                    for i in ridx:
                        rx += [0, pts2[i, 0], None]; ry += [0, pts2[i, 1], None]
                    fig.add_trace(go.Scatter(x=rx, y=ry, mode='lines', opacity=0.5, line=dict(color=colour, width=1),
                                             hoverinfo='skip', legendgroup=g, showlegend=False), row=k + 1, col=c)
                fig.add_trace(go.Scatter(
                    x=pts2[idx, 0], y=pts2[idx, 1], mode='markers',
                    marker=dict(size=[10 if i in crit else 6 for i in idx],
                                symbol=['diamond' if i in crit else 'circle' for i in idx], color=colour),
                    hovertext=_hover([names[i] for i in idx], pts3[idx]), hoverinfo='text',
                    name=_label(g), legendgroup=g, showlegend=(k == 0 and c == 1)), row=k + 1, col=c)
 
            fig.update_xaxes(range=[lo[ih], hi[ih]], title_text=label[ih] if k == 3 else None, row=k + 1, col=c)
            fig.update_yaxes(range=[hi[iv], lo[iv]] if rev else [lo[iv], hi[iv]],
                             title_text=label[iv], row=k + 1, col=c)
 
    fig.update_layout(height=1000, width=950, margin=dict(t=60),
                      title='Nodal load cloud: 2-D projections (diamond = on 3-D convex hull)')
    return fig