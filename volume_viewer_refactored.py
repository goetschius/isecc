import argparse
import time

import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from isecc import iseccFFT_v3

CMOCEAN_CMAPS = (
    "thermal",
    "haline",
    "solar",
    "ice",
    "gray",
    "oxy",
    "deep",
    "dense",
    "algae",
    "matter",
    "turbid",
    "speed",
    "amp",
    "tempo",
    "rain",
    "phase",
    "topo",
    "balance",
    "delta",
    "curl",
)

MATPLOTLIB_FALLBACK_CMAPS = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
)

AVAILABLE_CMAPS = CMOCEAN_CMAPS + MATPLOTLIB_FALLBACK_CMAPS


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
        choices=AVAILABLE_CMAPS,
        help=(
            "Colormap for radial coloring. Supports perceptually uniform cmocean "
            "maps plus a small matplotlib fallback set."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a coarse, faster render preset for troubleshooting.",
    )
    return parser.parse_args()


def get_colormap(cmap_name):
    if cmap_name in CMOCEAN_CMAPS:
        try:
            import cmocean
        except ImportError as exc:
            raise RuntimeError(
                f"Colormap '{cmap_name}' requires the cmocean package. "
                "Install it in this environment with `pip install cmocean`."
            ) from exc

        return getattr(cmocean.cm, cmap_name)

    return colormaps.get_cmap(cmap_name)


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


def resolve_angpix(cli_angpix, header_angpix):
    if cli_angpix is not None:
        angpix = cli_angpix
        source = "--angpix"
    elif header_angpix is not None:
        angpix = header_angpix
        source = "MRC header"
    else:
        angpix = 1.0
        source = "default"

    print(f"Using pixel size: {angpix:.4f} Å/pixel ({source})")
    return angpix


def extract_surface(ndimage, angpix, level=None, step_size=2, cmap_name="plasma"):
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

    colormap = get_colormap(cmap_name)
    facecolors = colormap(normalized)
    color_elapsed = time.perf_counter() - color_start
    print(f"Prepared mesh colors in {color_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    print(f"Surface extraction pipeline time: {total_elapsed:.2f}s")

    return verts_centered, faces, facecolors


def render_surface(
    verts_centered,
    faces,
    facecolors,
    output_file,
    elev=30.0,
    azim=45.0,
    edge_linewidth=0.05,
):
    render_start = time.perf_counter()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        verts_centered[faces],
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
    print(f"Rendered and saved image in {render_elapsed:.2f}s: {output_file}")


def main():
    args = parse_args()
    volume, header_angpix = load_volume(args.input_file)
    angpix = resolve_angpix(args.angpix, header_angpix)

    step_size = args.step_size
    level = args.level
    edge_linewidth = 0.05

    if args.fast:
        step_size = max(step_size, 4)
        if level is None:
            level = np.around((np.amax(volume) / 6.0) + 0.004, decimals=4)
        edge_linewidth = 0.0

    total_start = time.perf_counter()
    verts_centered, faces, facecolors = extract_surface(
        ndimage=volume,
        angpix=angpix,
        level=level,
        step_size=step_size,
        cmap_name=args.cmap,
    )
    render_surface(
        verts_centered=verts_centered,
        faces=faces,
        facecolors=facecolors,
        output_file=args.output_file,
        elev=args.elev,
        azim=args.azim,
        edge_linewidth=edge_linewidth,
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
