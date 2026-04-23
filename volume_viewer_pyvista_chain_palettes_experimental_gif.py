import argparse
from pathlib import Path
import shlex
import time

import mrcfile
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageSequence
from scipy import ndimage as ndi
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
DEFAULT_BASE_CMAP = "gray"
MAX_CHAIN_COLOR_GROUPS = 10
CHAIN_A_PALETTE = "YlOrRd"
CHAIN_HL_PALETTE = "PuBuGn"
CHAIN_A_LABEL = "Chain A"
CHAIN_HL_LABEL = "Chains H/L"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)
LIGHTING_PRESETS = ("flat", "balanced", "dramatic")
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
        description=(
            "Render an animated GIF isosurface view from an MRC volume using "
            "PyVista, with optional chain-aware coloring from an mmCIF model."
        )
    )
    parser.add_argument("input_file", nargs="?", help="Path to the input MRC file")
    parser.add_argument("output_file", nargs="?", help="Path to the output GIF file")
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="Print available palette names and exit.",
    )
    parser.add_argument(
        "--cif-file",
        default=str(Path(__file__).with_name("7M3N.cif")),
        help="Path to the mmCIF file used for chain-aware recoloring.",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=4.0,
        help="Distance in Angstroms used to recolor isosurface faces near chains.",
    )
    parser.add_argument(
        "--chain-palette",
        action="append",
        default=[],
        metavar="CHAINS:PALETTE",
        help=(
            "Assign a palette to one chain or a comma-separated chain group, for "
            "example 'A:YlOrRd' or 'H,L:PuBuGn'. Repeat up to 10 times."
        ),
    )
    parser.add_argument(
        "--symmetry-copy-chains",
        action="append",
        default=[],
        metavar="CHAINS",
        help=(
            "Generate icosahedral symmetry copies for the listed chain IDs before "
            "distance coloring, for example 'A' or 'H,L'. Repeat as needed."
        ),
    )
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
        "--hide-dust",
        action="store_true",
        help="Remove tiny disconnected above-threshold components before meshing.",
    )
    parser.add_argument(
        "--dust-volume-cutoff",
        type=float,
        default=4.0,
        help="Minimum connected-component volume in cubic Angstroms to keep when using --hide-dust.",
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
        default=DEFAULT_BASE_CMAP,
        choices=AVAILABLE_CMAPS,
        help=(
            "Base colormap for radial coloring outside selected chain neighborhoods. "
            "Supports perceptually uniform cmocean maps plus a small matplotlib "
            "fallback set."
        ),
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background color for the rendered GIF.",
    )
    parser.add_argument(
        "--lighting-preset",
        choices=LIGHTING_PRESETS,
        default="balanced",
        help="Experimental lighting preset for the rendered surface.",
    )
    parser.add_argument(
        "--smooth-shading",
        action="store_true",
        help="Use smooth shading instead of faceted shading.",
    )
    parser.add_argument(
        "--shadows",
        action="store_true",
        help="Enable renderer shadows when supported by the local PyVista/VTK build.",
    )
    parser.add_argument(
        "--silhouette",
        action="store_true",
        help="Add a silhouette overlay around the rendered surface.",
    )
    parser.add_argument(
        "--silhouette-color",
        default="black",
        help="Color used for the silhouette overlay.",
    )
    parser.add_argument(
        "--silhouette-width",
        type=float,
        default=None,
        help="Line width used for the silhouette overlay. Defaults to an auto-scaled value based on render size.",
    )
    parser.add_argument(
        "--outline",
        action="store_true",
        help="Add a wireframe outline overlay on top of the surface.",
    )
    parser.add_argument(
        "--outline-color",
        default="black",
        help="Color used for the outline overlay.",
    )
    parser.add_argument(
        "--outline-width",
        type=float,
        default=1.5,
        help="Line width used for the outline overlay.",
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
    parser.add_argument(
        "--scale-bar",
        action="store_true",
        help="Add a labeled scale bar in Angstroms to every GIF frame.",
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


def print_available_palettes():
    print("Available palettes:")
    for cmap_name in AVAILABLE_CMAPS:
        print(cmap_name)


def validate_required_paths(args):
    if args.list_palettes:
        return

    missing = []
    if not args.input_file:
        missing.append("input_file")
    if not args.output_file:
        missing.append("output_file")

    if missing:
        raise ValueError(
            "the following arguments are required unless --list-palettes is used: "
            + ", ".join(missing)
        )


def parse_chain_list(chain_text):
    chain_ids = [chain_id.strip() for chain_id in chain_text.split(",") if chain_id.strip()]
    if not chain_ids:
        raise ValueError("Expected at least one chain ID")
    return tuple(chain_ids)


def parse_chain_palette_specs(chain_palette_specs):
    if not chain_palette_specs:
        return [
            {
                "label": CHAIN_A_LABEL,
                "chain_ids": ("A",),
                "palette": CHAIN_A_PALETTE,
            },
            {
                "label": CHAIN_HL_LABEL,
                "chain_ids": ("H", "L"),
                "palette": CHAIN_HL_PALETTE,
            },
        ]

    if len(chain_palette_specs) > MAX_CHAIN_COLOR_GROUPS:
        raise ValueError(
            f"--chain-palette may be supplied at most {MAX_CHAIN_COLOR_GROUPS} times"
        )

    chain_groups = []
    assigned_chains = {}
    for spec in chain_palette_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --chain-palette value '{spec}'. Use CHAINS:PALETTE."
            )

        chain_text, palette_name = spec.split(":", 1)
        chain_ids = parse_chain_list(chain_text)
        palette_name = palette_name.strip()
        if not palette_name:
            raise ValueError(
                f"Invalid --chain-palette value '{spec}'. Palette name is required."
            )

        get_colormap(palette_name)

        duplicates = [chain_id for chain_id in chain_ids if chain_id in assigned_chains]
        if duplicates:
            raise ValueError(
                f"Chain IDs {duplicates} were assigned more than one palette group."
            )

        label = ",".join(chain_ids)
        chain_groups.append(
            {
                "label": label,
                "chain_ids": chain_ids,
                "palette": palette_name,
            }
        )

        for chain_id in chain_ids:
            assigned_chains[chain_id] = label

    return chain_groups


def parse_symmetry_copy_specs(symmetry_copy_specs):
    symmetry_chain_ids = set()
    for spec in symmetry_copy_specs:
        symmetry_chain_ids.update(parse_chain_list(spec))
    return symmetry_chain_ids


def load_volume(input_file):
    start_time = time.perf_counter()
    with mrcfile.open(input_file) as mrc:
        volume = np.array(mrc.data, copy=True)
        voxel_size = float(mrc.voxel_size.x) if mrc.voxel_size.x else None
        origin = np.array(
            [
                float(mrc.header.origin.x),
                float(mrc.header.origin.y),
                float(mrc.header.origin.z),
            ],
            dtype=np.float64,
        )
        starts = np.array(
            [
                float(mrc.header.nxstart),
                float(mrc.header.nystart),
                float(mrc.header.nzstart),
            ],
            dtype=np.float64,
        )

    volume = np.flipud(volume)
    volume = iseccFFT_v3.swapAxes_ndimage(volume)
    elapsed = time.perf_counter() - start_time
    print(f"Loaded volume in {elapsed:.2f}s: shape={volume.shape}, dtype={volume.dtype}")
    return volume, voxel_size, origin, starts


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


def resolve_world_origin(origin, starts, angpix):
    if np.any(np.abs(origin) > 1e-6):
        world_origin = origin
        source = "MRC header origin"
    elif np.any(np.abs(starts) > 1e-6):
        world_origin = starts * angpix
        source = "MRC nxstart/nystart/nzstart"
    else:
        world_origin = np.zeros(3, dtype=np.float64)
        source = "default"

    print(
        "Using world origin: "
        f"({world_origin[0]:.3f}, {world_origin[1]:.3f}, {world_origin[2]:.3f}) Å "
        f"({source})"
    )
    return world_origin


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


def remove_dust_components(ndimage, level, angpix, dust_volume_cutoff):
    voxel_volume = angpix ** 3
    min_voxels = max(1, int(np.ceil(dust_volume_cutoff / voxel_volume)))
    threshold_mask = ndimage >= level

    if not np.any(threshold_mask):
        print("No above-threshold voxels found for dust filtering")
        return ndimage

    structure = ndi.generate_binary_structure(3, 1)
    labels, num_labels = ndi.label(threshold_mask, structure=structure)
    if num_labels == 0:
        return ndimage

    component_sizes = np.bincount(labels.ravel())
    keep_component = component_sizes >= min_voxels
    keep_component[0] = False

    kept_count = int(np.count_nonzero(keep_component))
    removed_count = int(num_labels - kept_count)
    if removed_count == 0:
        print(
            f"Dust filter kept all {num_labels} above-threshold components "
            f"(cutoff {dust_volume_cutoff:.2f} A^3)"
        )
        return ndimage

    filtered = np.array(ndimage, copy=True)
    dust_mask = (labels > 0) & ~keep_component[labels]
    fill_value = min(float(np.min(ndimage)), float(level) - 1e-6)
    filtered[dust_mask] = fill_value
    print(
        f"Dust filter removed {removed_count} of {num_labels} components "
        f"smaller than {dust_volume_cutoff:.2f} A^3 "
        f"({min_voxels} voxels at {angpix:.4f} A/pixel)"
    )
    return filtered


def transformed_vertices_to_world(verts, volume_shape, angpix, world_origin):
    volume_shape = np.array(volume_shape, dtype=np.float64)
    x_coords = world_origin[0] + (verts[:, 0] * angpix)
    y_coords = world_origin[1] + (verts[:, 1] * angpix)
    z_coords = world_origin[2] + ((volume_shape[2] - 1.0 - verts[:, 2]) * angpix)
    return np.column_stack((x_coords, y_coords, z_coords))


def iterate_mmcif_loop_rows(cif_path, category_prefix):
    lines = Path(cif_path).read_text().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line != "loop_":
            i += 1
            continue

        i += 1
        headers = []
        while i < len(lines):
            header_line = lines[i].strip()
            if header_line.startswith("_"):
                headers.append(header_line)
                i += 1
                continue
            break

        if not headers or not all(header.startswith(category_prefix) for header in headers):
            continue

        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line or row_line == "#":
                break
            if row_line == "loop_" or row_line.startswith("_"):
                return

            parts = shlex.split(lines[i], posix=True)
            if len(parts) < len(headers):
                break

            yield dict(zip(headers, parts))
            i += 1
        return


def load_chain_coordinates_from_cif(cif_file, chain_ids):
    requested = set(chain_ids)
    atom_rows = iterate_mmcif_loop_rows(cif_file, "_atom_site.")
    chain_coords = {chain_id: [] for chain_id in requested}

    for row in atom_rows:
        chain_id = row.get("_atom_site.auth_asym_id")
        if chain_id not in requested:
            continue

        try:
            coords = (
                float(row["_atom_site.Cartn_x"]),
                float(row["_atom_site.Cartn_y"]),
                float(row["_atom_site.Cartn_z"]),
            )
        except (KeyError, ValueError):
            continue

        model_num = row.get("_atom_site.pdbx_PDB_model_num", "1")
        if model_num not in {"1", ".", "?"}:
            continue

        chain_coords[chain_id].append(coords)

    missing = [chain_id for chain_id, coords in chain_coords.items() if not coords]
    if missing:
        raise RuntimeError(
            f"Did not find atomic coordinates for chains {missing} in {cif_file}"
        )

    return {
        chain_id: np.asarray(coords, dtype=np.float64)
        for chain_id, coords in chain_coords.items()
    }


def get_all_symmetry_rotations():
    rotations = []
    for pyquat in symops.getSymOps():
        scipy_quat = iseccFFT_v3.pyquat2scipy(pyquat)
        rotations.append(R.from_quat(scipy_quat))
    return rotations


def expand_coordinates_with_symmetry(chain_coords, symmetry_chain_ids):
    if not symmetry_chain_ids:
        return chain_coords

    rotations = get_all_symmetry_rotations()
    expanded = {}
    for chain_id, coords in chain_coords.items():
        if chain_id not in symmetry_chain_ids:
            expanded[chain_id] = coords
            continue

        symmetry_copies = [rotation.apply(coords) for rotation in rotations]
        expanded[chain_id] = np.vstack(symmetry_copies)
        print(
            f"Expanded chain {chain_id} to {expanded[chain_id].shape[0]} atoms "
            f"across {len(rotations)} symmetry operators"
        )

    return expanded


def build_group_palette(normalized_values, palette_name):
    palette = get_colormap(palette_name)
    normalized = np.clip(normalized_values, 0.0, 1.0)
    return (palette(normalized)[:, :3] * 255).astype(np.uint8)


def recolor_faces_by_chain_proximity(
    centroids_world,
    base_facecolors,
    radial_normalized,
    chain_groups,
    chain_coords,
    cutoff,
):
    from scipy.spatial import cKDTree

    colors = np.array(base_facecolors, copy=True)
    group_distances = []
    for group in chain_groups:
        coords = np.vstack([chain_coords[chain_id] for chain_id in group["chain_ids"]])
        tree = cKDTree(coords)
        distances, _ = tree.query(
            centroids_world,
            k=1,
            distance_upper_bound=cutoff,
        )
        group_distances.append(distances)

    distance_matrix = np.vstack(group_distances)
    valid_matrix = np.isfinite(distance_matrix)
    any_match = np.any(valid_matrix, axis=0)
    nearest_group_indices = np.argmin(distance_matrix, axis=0)

    for group_index, group in enumerate(chain_groups):
        mask = any_match & (nearest_group_indices == group_index)
        if np.any(mask):
            colors[mask] = build_group_palette(
                radial_normalized[mask],
                palette_name=group["palette"],
            )

        print(
            f"Chains {','.join(group['chain_ids'])}: recolored "
            f"{int(np.count_nonzero(mask))} faces within {cutoff:.2f} Å using "
            f"{group['palette']}"
        )

    return colors


def extract_surface(
    ndimage,
    angpix,
    world_origin,
    cif_file=None,
    chain_groups=None,
    symmetry_chain_ids=None,
    distance_cutoff=4.0,
    hide_dust=False,
    dust_volume_cutoff=4.0,
    level=None,
    step_size=2,
    cmap_name=DEFAULT_BASE_CMAP,
):
    total_start = time.perf_counter()

    if level is None:
        level = np.around((np.amax(ndimage) / 10.0) + 0.004, decimals=4)

    if hide_dust:
        ndimage = remove_dust_components(
            ndimage=ndimage,
            level=level,
            angpix=angpix,
            dust_volume_cutoff=dust_volume_cutoff,
        )

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
    volume_shape = np.array(ndimage.shape, dtype=np.float64)
    center = (volume_shape - 1.0) / 2.0
    verts_centered = (verts - center) * angpix
    verts_world = transformed_vertices_to_world(
        verts=verts,
        volume_shape=volume_shape,
        angpix=angpix,
        world_origin=world_origin,
    )

    triangles_centered = verts_centered[faces]
    centroids_centered = triangles_centered.mean(axis=1)
    radii = np.linalg.norm(centroids_centered, axis=1)

    radius_span = radii.max() - radii.min()
    if radius_span == 0:
        normalized = np.zeros_like(radii)
    else:
        normalized = (radii - radii.min()) / radius_span

    colormap = get_colormap(cmap_name)
    facecolors = (colormap(normalized)[:, :3] * 255).astype(np.uint8)

    if cif_file is not None and chain_groups:
        requested_chain_ids = sorted(
            {
                chain_id
                for group in chain_groups
                for chain_id in group["chain_ids"]
            }
        )
        chain_coords = load_chain_coordinates_from_cif(
            cif_file=cif_file,
            chain_ids=requested_chain_ids,
        )
        chain_coords = expand_coordinates_with_symmetry(
            chain_coords=chain_coords,
            symmetry_chain_ids=symmetry_chain_ids or set(),
        )
        triangles_world = verts_world[faces]
        centroids_world = triangles_world.mean(axis=1)
        facecolors = recolor_faces_by_chain_proximity(
            centroids_world=centroids_world,
            base_facecolors=facecolors,
            radial_normalized=normalized,
            chain_groups=chain_groups,
            chain_coords=chain_coords,
            cutoff=distance_cutoff,
        )

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


def add_scale_bar_to_gif(gif_path, box_size_angstrom, background="white"):
    bar_length_angstrom = choose_scale_bar_length(box_size_angstrom)

    with Image.open(gif_path) as image:
        frames = []
        durations = []
        disposals = []

        for frame in ImageSequence.Iterator(image):
            rgba = frame.convert("RGBA")
            draw = ImageDraw.Draw(rgba)
            font = load_annotation_font(rgba.size[0])

            width, height = rgba.size
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

            frames.append(rgba)
            durations.append(frame.info.get("duration", image.info.get("duration", 40)))
            disposals.append(frame.disposal_method if hasattr(frame, "disposal_method") else 2)

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=image.info.get("loop", 0),
            disposal=disposals,
        )
        print(f"Added GIF scale bar: {bar_length_angstrom} Å")


def resolve_auto_silhouette_width(render_width, override_width=None):
    if override_width is not None:
        return override_width

    auto_width = render_width / 685.0
    auto_width = max(0.375, min(3.0, auto_width))
    print(
        f"Auto silhouette width: {auto_width:.2f} px for render width {render_width}"
    )
    return auto_width


def lighting_kwargs_for_preset(lighting_preset):
    if lighting_preset == "flat":
        return {
            "lighting": False,
            "ambient": 1.0,
            "diffuse": 0.0,
            "specular": 0.0,
            "specular_power": 1.0,
        }
    if lighting_preset == "dramatic":
        return {
            "lighting": True,
            "ambient": 0.18,
            "diffuse": 0.75,
            "specular": 0.35,
            "specular_power": 20.0,
        }

    return {
        "lighting": True,
        "ambient": 0.30,
        "diffuse": 0.65,
        "specular": 0.12,
        "specular_power": 10.0,
    }


def add_experimental_overlays(
    plotter,
    mesh,
    outline=False,
    outline_color="black",
    outline_width=1.5,
    silhouette=False,
    silhouette_color="black",
    silhouette_width=3.0,
):
    if outline:
        plotter.add_mesh(
            mesh.copy(deep=True),
            color=outline_color,
            opacity=0.0,
            lighting=False,
            silhouette={
                "color": outline_color,
                "line_width": outline_width,
            },
        )

    if silhouette:
        try:
            plotter.add_mesh(
                mesh.copy(deep=True),
                color=silhouette_color,
                opacity=0.0,
                lighting=False,
                silhouette={
                    "color": silhouette_color,
                    "line_width": silhouette_width,
                },
            )
        except Exception:
            try:
                silhouette_mesh = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
                if silhouette_mesh.n_cells > 0:
                    plotter.add_mesh(
                        silhouette_mesh,
                        color=silhouette_color,
                        line_width=silhouette_width,
                        lighting=False,
                    )
                else:
                    print("Silhouette overlay produced no visible boundary edges for this mesh")
            except Exception as exc:
                print(f"Silhouette overlay is unavailable in this environment: {exc}")


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
    scale_bar=False,
    box_size_angstrom=None,
    lighting_preset="balanced",
    smooth_shading=False,
    shadows=False,
    outline=False,
    outline_color="black",
    outline_width=1.5,
    silhouette=False,
    silhouette_color="black",
    silhouette_width=None,
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
    mesh_lighting = lighting_kwargs_for_preset(lighting_preset)
    silhouette_width = resolve_auto_silhouette_width(
        render_width=base_boxsize,
        override_width=silhouette_width,
    )

    plotter = pv.Plotter(off_screen=True, window_size=(base_boxsize, base_boxsize))
    plotter.set_background(background)
    if shadows and mesh_lighting["lighting"]:
        try:
            plotter.enable_shadows()
        except Exception as exc:
            print(f"Shadow rendering is unavailable in this environment: {exc}")
    plotter.add_mesh(
        mesh,
        scalars="face_rgb",
        rgb=True,
        show_edges=False,
        smooth_shading=smooth_shading,
        **mesh_lighting,
    )
    add_experimental_overlays(
        plotter=plotter,
        mesh=mesh,
        outline=outline,
        outline_color=outline_color,
        outline_width=outline_width,
        silhouette=silhouette,
        silhouette_color=silhouette_color,
        silhouette_width=silhouette_width,
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
    if scale_bar:
        add_scale_bar_to_gif(
            gif_path=output_file,
            box_size_angstrom=box_size_angstrom,
            background=background,
        )
    render_elapsed = time.perf_counter() - render_start
    print(f"Rendered and saved GIF in {render_elapsed:.2f}s: {output_file}")


def main():
    args = parse_args()
    validate_required_paths(args)
    if args.list_palettes:
        print_available_palettes()
        return

    chain_groups = parse_chain_palette_specs(args.chain_palette)
    symmetry_chain_ids = parse_symmetry_copy_specs(args.symmetry_copy_chains)
    volume, header_angpix, origin, starts = load_volume(args.input_file)
    angpix = resolve_angpix(args.angpix, header_angpix)
    world_origin = resolve_world_origin(origin, starts, angpix)
    box_size_angstrom = volume.shape[0] * angpix

    unknown_symmetry_chains = sorted(
        symmetry_chain_ids
        - {
            chain_id
            for group in chain_groups
            for chain_id in group["chain_ids"]
        }
    )
    if unknown_symmetry_chains:
        raise ValueError(
            "--symmetry-copy-chains referenced chains without a palette group: "
            f"{unknown_symmetry_chains}"
        )

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
        world_origin=world_origin,
        cif_file=args.cif_file,
        chain_groups=chain_groups,
        symmetry_chain_ids=symmetry_chain_ids,
        distance_cutoff=args.distance_cutoff,
        hide_dust=args.hide_dust,
        dust_volume_cutoff=args.dust_volume_cutoff,
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
        scale_bar=args.scale_bar,
        box_size_angstrom=box_size_angstrom,
        lighting_preset=args.lighting_preset,
        smooth_shading=args.smooth_shading,
        shadows=args.shadows,
        outline=args.outline,
        outline_color=args.outline_color,
        outline_width=args.outline_width,
        silhouette=args.silhouette,
        silhouette_color=args.silhouette_color,
        silhouette_width=args.silhouette_width,
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"Total render pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
