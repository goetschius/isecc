import argparse
import time

import mrcfile
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageColor, ImageSequence
from scipy.spatial.transform import Rotation as R
from skimage import measure

from isecc import iseccFFT_v3
from isecc import symops

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

QUALITY_PRESETS = {
    "fast": {"degrees_per_frame": 6.0, "fps": 15},
    "good": {"degrees_per_frame": 4.0, "fps": 18},
    "better": {"degrees_per_frame": 2.0, "fps": 20},
    "best": {"degrees_per_frame": 1.0, "fps": 24},
    "excellent": {"degrees_per_frame": 0.5, "fps": 24},
    "superb": {"degrees_per_frame": 0.25, "fps": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an animated GIF isosurface view from an MRC volume using PyVista."
    )
    parser.add_argument("input_file", help="Path to the input MRC file")
    parser.add_argument("output_file", help="Path to the output GIF file")
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
        default=20.0,
        help="Camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=0.0,
        help="Starting camera azimuth in degrees.",
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
        "--background",
        default="white",
        help="Background color for the rendered GIF.",
    )
    parser.add_argument(
        "--symop",
        type=int,
        default=1,
        help="Icosahedral symmetry operation to apply, numbered 1-60. 1 is identity.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a coarse, faster render preset for troubleshooting.",
    )
    parser.add_argument(
        "--quality",
        choices=tuple(QUALITY_PRESETS.keys()),
        default="better",
        help=(
            "Animation quality preset controlling rotation step and default FPS. "
            "Does not affect image size."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the output GIF. Overrides the --quality preset.",
    )
    parser.add_argument(
        "--base-boxsize",
        type=int,
        choices=(128, 256, 512, 1024),
        default=512,
        help="Square base render size in pixels before optional cropping.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Symmetrically crop away background whitespace after rendering.",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=0,
        help="Extra symmetric padding in pixels to keep when using --crop.",
    )
    parser.add_argument(
        "--crop-size-mode",
        choices=("multiple_of_10", "power_of_2"),
        default="multiple_of_10",
        help="Snap cropped square GIF size to a multiple of 10 or a power of 2.",
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


def resolve_animation_settings(quality, fps_override):
    settings = QUALITY_PRESETS[quality]
    degrees_per_frame = settings["degrees_per_frame"]
    fps = settings["fps"] if fps_override is None else fps_override
    num_frames = max(1, int(np.ceil(360.0 / degrees_per_frame)))
    print(
        f"Animation quality '{quality}': {degrees_per_frame:.2f} degrees/frame, "
        f"{num_frames} frames, {fps} fps"
    )
    return degrees_per_frame, fps, num_frames


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

    colormap = get_colormap(cmap_name)
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


def snap_square_size(size, limit, mode):
    if mode == "power_of_2":
        snapped = 2
        while snapped < size:
            snapped *= 2
    else:
        snapped = ((size + 9) // 10) * 10

    if snapped > limit:
        if mode == "power_of_2":
            snapped = 2 ** int(np.floor(np.log2(limit)))
        else:
            snapped = (limit // 10) * 10

    if snapped < size:
        snapped = size

    if snapped % 2 == 1:
        snapped += 1

    return min(snapped, limit)


def minimum_square_size_with_buffer(content_height, content_width, buffer_fraction=0.10):
    usable_fraction = 1.0 - (2.0 * buffer_fraction)
    if usable_fraction <= 0:
        raise ValueError("buffer_fraction is too large")

    content_size = max(content_height, content_width)
    return int(np.ceil(content_size / usable_fraction))


def estimate_frame_background_rgb(frame_array, fallback_background):
    return fallback_background


def crop_gif_symmetrically(gif_path, background="white", pad=0, tolerance=6, size_mode="multiple_of_10"):
    with Image.open(gif_path) as image:
        frames = []
        durations = []
        disposals = []
        fallback_background_rgb = np.array(ImageColor.getrgb(background), dtype=np.int16)

        global_top = None
        global_bottom = None
        global_left = None
        global_right = None

        for frame in ImageSequence.Iterator(image):
            rgba = frame.convert("RGBA")
            frames.append(rgba.copy())
            durations.append(frame.info.get("duration", image.info.get("duration", 40)))
            disposals.append(frame.disposal_method if hasattr(frame, "disposal_method") else 2)

            frame_array = np.array(rgba)
            background_rgb = estimate_frame_background_rgb(
                frame_array, fallback_background_rgb
            )
            rgb = frame_array[:, :, :3].astype(np.int16)
            alpha = frame_array[:, :, 3]
            color_distance = np.max(np.abs(rgb - background_rgb), axis=2)
            foreground_mask = (alpha > 0) & (color_distance > tolerance)

            if not np.any(foreground_mask):
                continue

            rows, cols = np.where(foreground_mask)
            top = int(rows.min())
            bottom = int(rows.max())
            left = int(cols.min())
            right = int(cols.max())

            global_top = top if global_top is None else min(global_top, top)
            global_bottom = bottom if global_bottom is None else max(global_bottom, bottom)
            global_left = left if global_left is None else min(global_left, left)
            global_right = right if global_right is None else max(global_right, right)

        if global_top is None:
            print(f"No foreground detected for crop: {gif_path}")
            return

        width, height = frames[0].size
        top = max(0, global_top - pad)
        left = max(0, global_left - pad)
        bottom = min(height - 1, global_bottom + pad)
        right = min(width - 1, global_right + pad)

        top_margin = top
        bottom_margin = height - 1 - bottom
        left_margin = left
        right_margin = width - 1 - right

        crop_y = min(top_margin, bottom_margin)
        crop_x = min(left_margin, right_margin)

        content_height = height - (2 * crop_y)
        content_width = width - (2 * crop_x)
        square_size = minimum_square_size_with_buffer(
            content_height=content_height,
            content_width=content_width,
            buffer_fraction=0.10,
        )
        square_size = snap_square_size(
            size=square_size,
            limit=min(height, width),
            mode=size_mode,
        )

        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        symmetric_top = int(round(center_y - (square_size / 2.0) + 0.5))
        symmetric_left = int(round(center_x - (square_size / 2.0) + 0.5))
        symmetric_top = max(0, min(symmetric_top, height - square_size))
        symmetric_left = max(0, min(symmetric_left, width - square_size))
        symmetric_bottom = symmetric_top + square_size
        symmetric_right = symmetric_left + square_size

        cropped_frames = [
            frame.crop((symmetric_left, symmetric_top, symmetric_right, symmetric_bottom))
            for frame in frames
        ]

        cropped_frames[0].save(
            gif_path,
            save_all=True,
            append_images=cropped_frames[1:],
            duration=durations,
            loop=image.info.get("loop", 0),
            disposal=disposals,
        )
        print(
            f"Cropped GIF symmetrically: {gif_path} -> "
            f"{square_size}x{square_size} ({size_mode})"
        )


def render_gif(
    verts_centered,
    faces,
    facecolors,
    output_file,
    elev=20.0,
    azim=0.0,
    background="white",
    fps=24,
    degrees_per_frame=2.0,
    crop=False,
    crop_pad=0,
    crop_size_mode="multiple_of_10",
    base_boxsize=512,
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

    plotter = pv.Plotter(off_screen=True, window_size=(base_boxsize, base_boxsize))
    plotter.set_background(background)
    plotter.add_mesh(
        mesh,
        scalars="face_rgb",
        rgb=True,
        show_edges=False,
        smooth_shading=False,
        lighting=True,
    )

    camera_distance = camera_distance_for_mesh(
        verts_centered,
        view_angle_degrees=plotter.camera.view_angle,
    )
    plotter.open_gif(output_file, fps=fps)

    frame_angles = np.arange(0.0, 360.0, degrees_per_frame, dtype=np.float64)
    for frame_azim in frame_angles:
        plotter.camera_position = camera_position_from_angles(
            elev=elev,
            azim=azim + float(frame_azim),
            distance=camera_distance,
        )
        plotter.camera.parallel_projection = False
        plotter.reset_camera_clipping_range()
        plotter.write_frame()

    plotter.close()
    if crop:
        crop_gif_symmetrically(
            gif_path=output_file,
            background=background,
            pad=crop_pad,
            size_mode=crop_size_mode,
        )
    render_elapsed = time.perf_counter() - render_start
    print(f"Rendered and saved GIF in {render_elapsed:.2f}s: {output_file}")


def main():
    args = parse_args()
    volume, header_angpix = load_volume(args.input_file)
    angpix = resolve_angpix(args.angpix, header_angpix)

    if not args.crop and args.crop_size_mode != "multiple_of_10":
        print(
            "Warning: --crop-size-mode was supplied without --crop; "
            "crop sizing will not be applied."
        )

    step_size = args.step_size
    level = args.level

    if args.fast:
        step_size = max(step_size, 4)
        if level is None:
            level = np.around((np.amax(volume) / 6.0) + 0.004, decimals=4)

    degrees_per_frame, fps, _ = resolve_animation_settings(args.quality, args.fps)

    total_start = time.perf_counter()
    verts_centered, faces, facecolors = extract_surface(
        ndimage=volume,
        angpix=angpix,
        level=level,
        step_size=step_size,
        cmap_name=args.cmap,
    )
    verts_centered = apply_symmetry_operation(verts_centered, args.symop)
    render_gif(
        verts_centered=verts_centered,
        faces=faces,
        facecolors=facecolors,
        output_file=args.output_file,
        elev=args.elev,
        azim=args.azim,
        background=args.background,
        fps=fps,
        degrees_per_frame=degrees_per_frame,
        crop=args.crop,
        crop_pad=args.crop_pad,
        crop_size_mode=args.crop_size_mode,
        base_boxsize=args.base_boxsize,
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
