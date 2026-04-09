import argparse
from pathlib import Path
import time

import mrcfile
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageColor, ImageDraw, ImageFont
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

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


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
    parser.add_argument(
        "--all-cmaps",
        action="store_true",
        help="Render one output for every available colormap using the output path as a filename stem.",
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
        choices=("multiple_of_100", "power_of_2"),
        default="multiple_of_100",
        help="Snap cropped square output size to a multiple of 100 or a power of 2.",
    )
    parser.add_argument(
        "--scale-bar",
        action="store_true",
        help="Add a labeled scale bar in Angstroms to the output image.",
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


def output_path_for_cmap(output_file, cmap_name):
    output_path = Path(output_file)
    suffix = output_path.suffix
    stem = output_path.stem
    parent = output_path.parent
    return parent / f"{stem}_cmap-{cmap_name}{suffix}"


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
        snapped = ((size + 99) // 100) * 100

    if snapped > limit:
        if mode == "power_of_2":
            snapped = 2 ** int(np.floor(np.log2(limit)))
        else:
            snapped = (limit // 100) * 100

    if snapped < size:
        snapped = size

    if snapped % 2 == 1:
        snapped += 1

    return min(snapped, limit)


def choose_scale_bar_length(box_size_angstrom):
    target = box_size_angstrom * 0.25
    candidates = [25, 50]

    max_multiple = max(1, int(np.ceil(box_size_angstrom / 100.0)))
    for multiplier in range(1, max_multiple + 1):
        candidates.append(multiplier * 100)

    candidates = sorted(set(candidate for candidate in candidates if candidate <= box_size_angstrom))
    if not candidates:
        return min(25, box_size_angstrom)

    return min(candidates, key=lambda candidate: abs(candidate - target))


def load_annotation_font(image_width):
    font_size = max(18, image_width // 28)

    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def crop_image_symmetrically(
    image_path,
    background="white",
    pad=0,
    tolerance=6,
    size_mode="multiple_of_100",
):
    with Image.open(image_path) as image:
        image = image.convert("RGBA")
        image_array = np.array(image)

        background_rgb = np.array(ImageColor.getrgb(background), dtype=np.int16)
        rgb = image_array[:, :, :3].astype(np.int16)
        alpha = image_array[:, :, 3]

        color_distance = np.max(np.abs(rgb - background_rgb), axis=2)
        foreground_mask = (alpha > 0) & (color_distance > tolerance)

        if not np.any(foreground_mask):
            print(f"No foreground detected for crop: {image_path}")
            return

        rows, cols = np.where(foreground_mask)
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(cols.min())
        right = int(cols.max())

        height, width = foreground_mask.shape
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(height - 1, bottom + pad)
        right = min(width - 1, right + pad)

        top_margin = top
        bottom_margin = height - 1 - bottom
        left_margin = left
        right_margin = width - 1 - right

        crop_y = min(top_margin, bottom_margin)
        crop_x = min(left_margin, right_margin)

        content_height = height - (2 * crop_y)
        content_width = width - (2 * crop_x)
        square_size = max(content_height, content_width)
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
        symmetric_bottom = symmetric_top + square_size - 1
        symmetric_right = symmetric_left + square_size - 1

        cropped = image.crop(
            (
                symmetric_left,
                symmetric_top,
                symmetric_right + 1,
                symmetric_bottom + 1,
            )
        )
        cropped.save(image_path)
        print(
            f"Cropped image symmetrically: {image_path} -> "
            f"{cropped.size[0]}x{cropped.size[1]} ({size_mode})"
        )


def add_scale_bar(image_path, box_size_angstrom, background="white"):
    bar_length_angstrom = choose_scale_bar_length(box_size_angstrom)

    with Image.open(image_path) as image:
        image = image.convert("RGBA")
        draw = ImageDraw.Draw(image)
        font = load_annotation_font(image.size[0])

        width, height = image.size
        pixels_per_angstrom = width / float(box_size_angstrom)
        bar_length_pixels = max(1, int(round(bar_length_angstrom * pixels_per_angstrom)))

        margin = max(12, width // 20)
        bar_thickness = max(4, width // 200)
        text_gap = max(8, width // 80)
        text = f"{int(bar_length_angstrom)} Å"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        bar_x0 = width - margin - bar_length_pixels
        bar_x1 = width - margin
        bar_y1 = height - margin
        bar_y0 = bar_y1 - bar_thickness
        text_x = bar_x0 + (bar_length_pixels - text_width) // 2
        text_y = bar_y0 - text_gap - text_height

        bg_rgb = ImageColor.getrgb(background)
        brightness = sum(bg_rgb) / 3.0
        foreground = (0, 0, 0, 255) if brightness > 127 else (255, 255, 255, 255)
        outline = (255, 255, 255, 255) if brightness > 127 else (0, 0, 0, 255)

        draw.rectangle(
            [bar_x0, bar_y0, bar_x1, bar_y1],
            fill=foreground,
            outline=outline,
            width=1,
        )
        draw.text(
            (text_x, text_y),
            text,
            fill=foreground,
            font=font,
            stroke_width=1,
            stroke_fill=outline,
        )

        image.save(image_path)
        print(f"Added scale bar: {bar_length_angstrom} Å")


def render_surface(
    verts_centered,
    faces,
    facecolors,
    output_file,
    elev=30.0,
    azim=45.0,
    background="white",
    window_size=(1600, 1600),
    crop=False,
    crop_pad=0,
    crop_size_mode="multiple_of_100",
    scale_bar=False,
    box_size_angstrom=None,
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
        show_edges=False,
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

    if crop:
        crop_image_symmetrically(
            image_path=output_file,
            background=background,
            pad=crop_pad,
            size_mode=crop_size_mode,
        )

    if scale_bar:
        add_scale_bar(
            image_path=output_file,
            box_size_angstrom=box_size_angstrom,
            background=background,
        )

    render_elapsed = time.perf_counter() - render_start
    print(f"Rendered and saved image in {render_elapsed:.2f}s: {output_file}")


def main():
    args = parse_args()
    volume, header_angpix = load_volume(args.input_file)
    angpix = args.angpix if args.angpix is not None else (header_angpix or 1.0)
    box_size_angstrom = volume.shape[0] * angpix

    step_size = args.step_size
    level = args.level

    if args.fast:
        step_size = max(step_size, 4)
        if level is None:
            level = np.around((np.amax(volume) / 6.0) + 0.004, decimals=4)

    total_start = time.perf_counter()
    cmap_names = AVAILABLE_CMAPS if args.all_cmaps else [args.cmap]
    symop_indices = range(1, 61) if args.all_symops else [args.symop]

    for cmap_name in cmap_names:
        verts_centered, faces, facecolors = extract_surface(
            ndimage=volume,
            angpix=angpix,
            level=level,
            step_size=step_size,
            cmap_name=cmap_name,
        )

        for symop_index in symop_indices:
            rotated_verts = apply_symmetry_operation(verts_centered, symop_index)

            output_file = Path(args.output_file)
            if args.all_cmaps:
                output_file = output_path_for_cmap(output_file, cmap_name)
            if args.all_symops:
                output_file = output_path_for_symop(output_file, symop_index)

            render_surface(
                verts_centered=rotated_verts,
                faces=faces,
                facecolors=facecolors,
                output_file=str(output_file),
                elev=args.elev,
                azim=args.azim,
                background=args.background,
                window_size=tuple(args.window_size),
                crop=args.crop,
                crop_pad=args.crop_pad,
                crop_size_mode=args.crop_size_mode,
                scale_bar=args.scale_bar,
                box_size_angstrom=box_size_angstrom,
            )

    total_elapsed = time.perf_counter() - total_start
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
