#!/usr/bin/env python3
"""
Command-line entry for visualizing the Birkhoff polytope.

This module is intended to be installed as a small CLI tool or run via
`python -m birkhoff_polytope_demo.visualize_birkhoff`.
"""
from __future__ import annotations

import argparse
import itertools
import os
import tempfile
import webbrowser
from typing import List

import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from plotly.colors import qualitative


def permutation_matrices(n: int) -> np.ndarray:
    mats = []
    for p in itertools.permutations(range(n)):
        M = np.zeros((n, n), dtype=float)
        for i, j in enumerate(p):
            M[i, j] = 1.0
        mats.append(M)
    return np.array(mats)


def pca_reduce(X: np.ndarray, k: int = 3) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    components = VT[:k]
    return Xc.dot(components.T)


def permutation_to_string(M: np.ndarray) -> str:
    rows = [" ".join(str(int(x)) for x in row) for row in M]
    return "<br>".join(rows)


def visualize(n: int = 3, out_html: str | None = None, open_browser: bool = True) -> str:
    mats = permutation_matrices(n)
    verts = mats.reshape((mats.shape[0], -1))
    pts3 = pca_reduce(verts, 3)

    hull = ConvexHull(pts3)

    i = hull.simplices[:, 0]
    j = hull.simplices[:, 1]
    k = hull.simplices[:, 2]

    mesh = go.Mesh3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        i=i, j=j, k=k,
        opacity=0.45,
        color='lightblue',
        name='hull'
    )

    labels = [f"P{idx}" for idx in range(len(pts3))]
    hovertext = [permutation_to_string(M) for M in mats]

    palette = qualitative.Plotly
    colors = [palette[idx % len(palette)] for idx in range(len(pts3))]

    scatter = go.Scatter3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        mode='markers+text',
        marker=dict(size=6, color=colors),
        text=labels,
        textposition='top center',
        hovertext=hovertext,
        hoverinfo='text',
        name='permutation vertices'
    )

    fig = go.Figure(data=[mesh, scatter])
    fig.update_layout(
        title=f'Birkhoff polytope (n={n}) — permutation matrices projected to 3D',
        scene=dict(aspectmode='auto')
    )

    if out_html is None:
        outputs_dir = os.path.join(os.getcwd(), 'birkhoff_outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        out_html = os.path.join(outputs_dir, f'birkhoff_n{n}.html')

    fig.write_html(out_html, auto_open=False)
    if open_browser:
        webbrowser.open('file://' + os.path.realpath(out_html))
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize Birkhoff polytope (convex hull of permutation matrices)')
    parser.add_argument('--n', type=int, default=3, choices=[2, 3, 4],
                        help='matrix size n (recommended 3 or 4)')
    parser.add_argument('--out', dest='out', type=str, default=None, help='output HTML file path')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false', help='do not open browser automatically')
    args = parser.parse_args()

    out = visualize(n=args.n, out_html=args.out, open_browser=args.open_browser)
    print('Wrote interactive plot to:', out)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Visualize the Birkhoff polytope (convex hull of permutation matrices).

This script generates all n x n permutation matrices, flattens them,
reduces to 3D with PCA (SVD), computes the convex hull and shows
an interactive 3D plot (opens in your browser).

Usage:
  python3 visualize_birkhoff.py --n 3
"""
import argparse
import itertools
import tempfile
import webbrowser
import os

import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go


def permutation_matrices(n):
    mats = []
    for p in itertools.permutations(range(n)):
        M = np.zeros((n, n), dtype=float)
        for i, j in enumerate(p):
            M[i, j] = 1.0
        mats.append(M)
    return np.array(mats)


def pca_reduce(X, k=3):
    Xc = X - X.mean(axis=0)
    # Compute SVD and take top-k right singular vectors (components)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    components = VT[:k]  # shape (k, features)
    return Xc.dot(components.T)


def visualize(n=3, out_html=None, open_browser=True):
    mats = permutation_matrices(n)
    verts = mats.reshape((mats.shape[0], -1))
    pts3 = pca_reduce(verts, 3)

    hull = ConvexHull(pts3)

    # ConvexHull.simplices gives triangles (indices into pts3)
    i = hull.simplices[:, 0]
    j = hull.simplices[:, 1]
    k = hull.simplices[:, 2]

    mesh = go.Mesh3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        i=i, j=j, k=k,
        opacity=0.45,
        color='lightblue',
        name='hull'
    )

    labels = [f"P{idx}" for idx in range(len(pts3))]
    # Build hover text showing the permutation matrix for each vertex
    hovertext = []
    for M in mats:
        rows = [' '.join(str(int(x)) for x in row) for row in M]
        hovertext.append('<br>'.join(rows))

    scatter = go.Scatter3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=labels,
        textposition='top center',
        hovertext=hovertext,
        hoverinfo='text',
        name='permutation vertices'
    )

    fig = go.Figure(data=[mesh, scatter])
    fig.update_layout(
        title=f'Birkhoff polytope (n={n}) — permutation matrices projected to 3D',
        scene=dict(aspectmode='auto')
    )

    if out_html is None:
        fd = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        out_html = fd.name
        fd.close()

    fig.write_html(out_html, auto_open=False)
    print('Wrote interactive plot to:', out_html)
    if open_browser:
        webbrowser.open('file://' + os.path.realpath(out_html))


def main():
    parser = argparse.ArgumentParser(description='Visualize Birkhoff polytope (convex hull of permutation matrices)')
    parser.add_argument('--n', type=int, default=3, choices=[2, 3, 4],
                        help='matrix size n (recommended 3 or 4)')
    parser.add_argument('--out', type=str, default=None, help='output HTML file path')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false', help='do not open browser automatically')
    args = parser.parse_args()

    visualize(n=args.n, out_html=args.out, open_browser=args.open_browser)


if __name__ == '__main__':
    main()
