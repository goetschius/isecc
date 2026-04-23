import argparse
from pathlib import Path
import time

import mrcfile
import numpy as np
from matplotlib import colormaps
from scipy.spatial.transform import Rotation as R
from skimage import measure

from isecc import iseccFFT_v3
from isecc import symops


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an isosurface view from an MRC volume using PyVista."
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
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1600, 1600),
        help="Off-screen render size in pixels.",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background color for the rendered image.",
    )
    parser.add_argument(
        "--symop",
        type=int,
        default=1,
        help="Icosahedral symmetry operation to apply, numbered 1-60. 1 is identity.",
    )
    parser.add_argument(
        "--all-symops",
        action="store_true",
        help="Render all 60 symmetry operations using the output path as a filename stem.",
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
    centroids = triangles.mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)

    radius_span = radii.max() - radii.min()
    if radius_span == 0:
        normalized = np.zeros_like(radii)
    else:
        normalized = (radii - radii.min()) / radius_span

    colormap = colormaps.get_cmap(cmap_name)
    facecolors = (colormap(normalized)[:, :3] * 255).astype(np.uint8)

    color_elapsed = time.perf_counter() - color_start
    total_elapsed = time.perf_counter() - total_start
    print(f"Prepared mesh colors in {color_elapsed:.2f}s")
    print(f"Surface extraction pipeline time: {total_elapsed:.2f}s")

    return verts_centered, faces, facecolors


def make_pyvista_mesh(verts_centered, faces, facecolors):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is not installed in this Python environment. "
            "Install it with `pip install pyvista` and rerun this script."
        ) from exc

    # PolyData expects each face prefixed by its vertex count.
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()

    mesh = pv.PolyData(verts_centered, faces_pv)
    mesh.cell_data["face_rgb"] = facecolors
    return mesh


def apply_symmetry_operation(verts_centered, symop_index):
    if not 1 <= symop_index <= 60:
        raise ValueError(f"--symop must be between 1 and 60, got {symop_index}")

    quaternions = symops.getSymOps()
    pyquat = quaternions[symop_index - 1]
    scipy_quat = iseccFFT_v3.pyquat2scipy(pyquat)
    rotation = R.from_quat(scipy_quat)

    rotated_verts = rotation.apply(verts_centered)
    print(f"Applied symmetry operation {symop_index}: {pyquat.tolist()}")
    return rotated_verts


def output_path_for_symop(output_file, symop_index):
    output_path = Path(output_file)
    suffix = output_path.suffix
    stem = output_path.stem
    parent = output_path.parent
    return parent / f"{stem}_symop{symop_index:02d}{suffix}"


def camera_position_from_angles(elev, azim, distance):
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)

    position = np.array(
        [
            distance * np.cos(elev_rad) * np.cos(azim_rad),
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.sin(elev_rad),
        ]
    )

    return [tuple(position), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def camera_distance_for_mesh(verts_centered, view_angle_degrees=30.0, padding=1.15):
    mins = verts_centered.min(axis=0)
    maxs = verts_centered.max(axis=0)
    diagonal = np.linalg.norm(maxs - mins)

    if diagonal == 0:
        return 1.0

    half_diagonal = diagonal / 2.0
    half_fov_radians = np.deg2rad(view_angle_degrees / 2.0)
    return (half_diagonal / np.tan(half_fov_radians)) * padding


def render_surface(
    verts_centered,
    faces,
    facecolors,
    output_file,
    elev=30.0,
    azim=45.0,
    background="white",
    window_size=(1600, 1600),
):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is not installed in this Python environment. "
            "Install it with `pip install pyvista` and rerun this script."
        ) from exc

    render_start = time.perf_counter()
    mesh = make_pyvista_mesh(verts_centered, faces, facecolors)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(
        mesh,
        scalars="face_rgb",
        rgb=True,
        show_edges=True,
        smooth_shading=False,
        lighting=True,
    )

    camera_distance = camera_distance_for_mesh(
        verts_centered,
        view_angle_degrees=plotter.camera.view_angle,
    )
    plotter.camera_position = camera_position_from_angles(
        elev, azim, camera_distance
    )
    plotter.camera.parallel_projection = False
    plotter.reset_camera_clipping_range()
    plotter.show(screenshot=output_file, auto_close=False)
    plotter.close()

    render_elapsed = time.perf_counter() - render_start
    print(f"Rendered and saved image in {render_elapsed:.2f}s: {output_file}")


def main():
    args = parse_args()
    volume, header_angpix = load_volume(args.input_file)
    angpix = args.angpix if args.angpix is not None else (header_angpix or 1.0)

    step_size = args.step_size
    level = args.level

    if args.fast:
        step_size = max(step_size, 4)
        if level is None:
            level = np.around((np.amax(volume) / 6.0) + 0.004, decimals=4)

    total_start = time.perf_counter()
    verts_centered, faces, facecolors = extract_surface(
        ndimage=volume,
        angpix=angpix,
        level=level,
        step_size=step_size,
        cmap_name=args.cmap,
    )
    symop_indices = range(1, 61) if args.all_symops else [args.symop]

    for symop_index in symop_indices:
        rotated_verts = apply_symmetry_operation(verts_centered, symop_index)
        output_file = (
            output_path_for_symop(args.output_file, symop_index)
            if args.all_symops
            else args.output_file
        )
        render_surface(
            verts_centered=rotated_verts,
            faces=faces,
            facecolors=facecolors,
            output_file=str(output_file),
            elev=args.elev,
            azim=args.azim,
            background=args.background,
            window_size=tuple(args.window_size),
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
