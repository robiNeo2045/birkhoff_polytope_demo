# Birkhoff Polytope Visualization

This small tool visualizes the Birkhoff polytope (the convex hull of permutation matrices) by projecting permutation matrices into 3D and rendering the convex hull.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


2. Run the visualizer (recommended `n=3` or `n=4`):

Run directly with the module:

```bash
python -m birkhoff_polytope_demo.visualize_birkhoff --n 3
```

Or install the package locally and use the console script:

```bash
pip install -e .
birkhoff-visualize --n 3
```

The script will open an interactive HTML plot in your browser by default. Use `--out path.html` to save to a specific file and `--no-browser` to prevent automatically opening the browser. Output files default to `./birkhoff_outputs/birkhoff_n{n}.html`.

**Demo & GitHub Pages**

- A generated demo HTML for `n=4` is available in this repository under `docs/birkhoff_n4.html` and will be deployed to GitHub Pages shortly after this commit (if enabled).
- Release page with attached HTML artifact: https://github.com/robiNeo2045/birkhoff_polytope_demo/releases/tag/v0.1.0


Notes
- For `n=3` there are 6 vertices (permutation matrices). For `n=4` there are 24 vertices.
- The script projects the high-dimensional vertices to 3D using PCA (SVD) before computing the convex hull for visualization.
