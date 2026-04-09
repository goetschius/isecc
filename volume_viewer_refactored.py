import argparse
import time

import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from isecc import iseccFFT_v3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an isosurface view from an MRC volume efficiently."
    )
    parser.add_argument("input_file", help="Path to the input MRC file")
    parser.add_argument("output_file", help="Path to the output image file")
    parser.add_argument(
        "--angpix",
        type=float,
        default=None,
        help="Pixel size in Angstroms. Defaults to the value stored in the MRC header.",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        help="Isosurface threshold. Defaults to max(volume) / 10 + 0.004.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=2,
        help="Marching cubes step size. Larger values are faster but less detailed.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=45.0,
        help="Camera azimuth in degrees.",
    )
    parser.add_argument(
        "--cmap",
        default="plasma",
        help="Matplotlib colormap name for radial coloring.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a coarse, faster render preset for troubleshooting.",
    )
    return parser.parse_args()


def load_volume(input_file):
    start_time = time.perf_counter()
    with mrcfile.open(input_file) as mrc:
        volume = np.array(mrc.data, copy=True)
        voxel_size = float(mrc.voxel_size.x) if mrc.voxel_size.x else None

    volume = np.flipud(volume)
    volume = iseccFFT_v3.swapAxes_ndimage(volume)
    elapsed = time.perf_counter() - start_time
    print(f"Loaded volume in {elapsed:.2f}s: shape={volume.shape}, dtype={volume.dtype}")
    return volume, voxel_size


def render_isosurface(
    ndimage,
    output_file,
    angpix,
    level=None,
    step_size=2,
    elev=30.0,
    azim=45.0,
    cmap_name="plasma",
    edge_linewidth=0.05,
):
    total_start = time.perf_counter()

    if level is None:
        level = np.around((np.amax(ndimage) / 10.0) + 0.004, decimals=4)

    marching_start = time.perf_counter()
    verts, faces, _, _ = measure.marching_cubes(
        ndimage, level=level, step_size=step_size
    )
    marching_elapsed = time.perf_counter() - marching_start
    print(
        f"Extracted isosurface in {marching_elapsed:.2f}s: "
        f"level={level}, step_size={step_size}, verts={len(verts)}, faces={len(faces)}"
    )

    color_start = time.perf_counter()
    center = (np.array(ndimage.shape, dtype=np.float64) - 1.0) / 2.0
    verts_centered = (verts - center) * angpix

    triangles = verts_centered[faces]

    # Color each face by the radius of its centroid from the volume center.
    centroids = triangles.mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)
    radius_span = radii.max() - radii.min()
    if radius_span == 0:
        normalized = np.zeros_like(radii)
    else:
        normalized = (radii - radii.min()) / radius_span

    colormap = colormaps.get_cmap(cmap_name)
    facecolors = colormap(normalized)
    color_elapsed = time.perf_counter() - color_start
    print(f"Prepared mesh colors in {color_elapsed:.2f}s")

    render_start = time.perf_counter()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors="k",
        linewidths=edge_linewidth,
    )
    ax.add_collection3d(mesh)

    maxrad = np.max(np.abs(verts_centered))
    ax.set_xlim(-maxrad, maxrad)
    ax.set_ylim(-maxrad, maxrad)
    ax.set_zlim(-maxrad, maxrad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    render_elapsed = time.perf_counter() - render_start
    total_elapsed = time.perf_counter() - total_start
    print(f"Rendered and saved image in {render_elapsed:.2f}s: {output_file}")
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


def main():
    args = parse_args()
    volume, header_angpix = load_volume(args.input_file)
    angpix = args.angpix if args.angpix is not None else (header_angpix or 1.0)

    step_size = args.step_size
    level = args.level
    edge_linewidth = 0.05

    if args.fast:
        step_size = max(step_size, 4)
        if level is None:
            level = np.around((np.amax(volume) / 6.0) + 0.004, decimals=4)
        edge_linewidth = 0.0

    render_isosurface(
        ndimage=volume,
        output_file=args.output_file,
        angpix=angpix,
        level=level,
        step_size=step_size,
        elev=args.elev,
        azim=args.azim,
        cmap_name=args.cmap,
        edge_linewidth=edge_linewidth,
    )


if __name__ == "__main__":
    main()
