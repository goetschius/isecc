#!/usr/bin/env python3
"""Keep one low-damage cyclic sector from an MRC mask.

Prototype CLI for cutting a Cn wedge around the Z axis.  The two cut
boundaries are separated by 360/n degrees and are allowed to follow local
low-mask valleys as a function of radius.
"""

import argparse
import json
import math
from pathlib import Path

import mrcfile
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
except ImportError:  # pragma: no cover - only used on very old installs
    gaussian_filter = None
    gaussian_filter1d = None


def parse_order(value):
    text = str(value).strip().upper()
    if text.startswith("C"):
        text = text[1:]
    try:
        order = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a cyclic symmetry like C12 or an integer, got {value!r}"
        ) from exc
    if order < 2:
        raise argparse.ArgumentTypeError("Cyclic order must be 2 or greater")
    return order


def circular_delta_bins(a, b, ntheta):
    return ((a - b + (ntheta // 2)) % ntheta) - (ntheta // 2)


def load_mrc(path):
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        voxel_size = (
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z),
        )
        origin = (
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z),
        )
        starts = (
            int(mrc.header.nxstart),
            int(mrc.header.nystart),
            int(mrc.header.nzstart),
        )
    return data, voxel_size, origin, starts


def write_mrc(path, data, voxel_size, origin, starts):
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.header.nxstart = starts[0]
        mrc.header.nystart = starts[1]
        mrc.header.nzstart = starts[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()


def polar_maps(shape, radial_bins, angular_bins):
    ny, nx = shape[1], shape[2]
    center_y = (ny - 1.0) / 2.0
    center_x = (nx - 1.0) / 2.0
    yy, xx = np.meshgrid(
        np.arange(ny, dtype=np.float32) - center_y,
        np.arange(nx, dtype=np.float32) - center_x,
        indexing="ij",
    )
    radius = np.sqrt((xx * xx) + (yy * yy))
    max_radius = float(np.max(radius))
    theta = np.mod(np.arctan2(yy, xx), 2.0 * math.pi)
    radial_bin = np.floor(radius / max_radius * (radial_bins - 1)).astype(np.int32)
    angular_bin = np.floor(theta / (2.0 * math.pi) * angular_bins).astype(np.int32)
    radial_bin = np.clip(radial_bin, 0, radial_bins - 1)
    angular_bin = np.mod(angular_bin, angular_bins)
    return radius, theta, radial_bin, angular_bin, max_radius


def build_cost_map(volume, radial_bin, angular_bin, radial_bins, angular_bins):
    flat_index = (radial_bin.ravel() * angular_bins) + angular_bin.ravel()
    cost = np.zeros(radial_bins * angular_bins, dtype=np.float64)
    for z_index in range(volume.shape[0]):
        cost += np.bincount(
            flat_index,
            weights=np.maximum(volume[z_index].ravel(), 0.0),
            minlength=radial_bins * angular_bins,
        )
    return cost.reshape(radial_bins, angular_bins)


def smooth_cost(cost, radial_sigma, angular_sigma):
    if gaussian_filter is None:
        return cost
    return gaussian_filter(
        cost,
        sigma=(float(radial_sigma), float(angular_sigma)),
        mode=("nearest", "wrap"),
    )


def choose_nominal_boundary(cost, order, min_radial_bin):
    profile = np.sum(cost[min_radial_bin:, :], axis=0)
    if gaussian_filter1d is not None:
        profile = gaussian_filter1d(profile, sigma=2.0, mode="wrap")
    angular_bins = cost.shape[1]
    sector_bins = int(round(angular_bins / order))
    scores = profile + np.roll(profile, -sector_bins)
    start_bin = int(np.argmin(scores))
    return start_bin, (start_bin + sector_bins) % angular_bins, float(scores[start_bin])


def follow_valley(cost, nominal_bin, search_bins, smoothness):
    radial_bins, angular_bins = cost.shape
    offsets = np.arange(-search_bins, search_bins + 1, dtype=np.int32)
    candidates = np.mod(nominal_bin + offsets, angular_bins)
    n_candidates = len(candidates)

    dp = np.empty((radial_bins, n_candidates), dtype=np.float64)
    back = np.zeros((radial_bins, n_candidates), dtype=np.int32)
    dp[0] = cost[0, candidates]

    for r_index in range(1, radial_bins):
        local = cost[r_index, candidates]
        previous = dp[r_index - 1]
        for c_index, candidate in enumerate(candidates):
            deltas = circular_delta_bins(candidate, candidates, angular_bins)
            penalties = smoothness * (deltas.astype(np.float64) ** 2)
            best_previous = int(np.argmin(previous + penalties))
            dp[r_index, c_index] = local[c_index] + previous[best_previous] + penalties[best_previous]
            back[r_index, c_index] = best_previous

    path = np.empty(radial_bins, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for r_index in range(radial_bins - 2, -1, -1):
        path[r_index] = back[r_index + 1, path[r_index + 1]]
    return candidates[path]


def path_to_relative_angles(path_bins, center_angle, angular_bins):
    angles = (path_bins.astype(np.float64) + 0.5) / angular_bins * (2.0 * math.pi)
    return (angles - center_angle + math.pi) % (2.0 * math.pi) - math.pi


def make_sector_selector(
    theta,
    radial_bin,
    lower_nominal_bin,
    lower_path,
    upper_path,
    order,
    angular_bins,
    edge_softness_degrees,
    axis_radius_pixels,
    radius,
):
    lower0 = (lower_nominal_bin + 0.5) / angular_bins * (2.0 * math.pi)
    sector_width = 2.0 * math.pi / order
    center_angle = (lower0 + (sector_width / 2.0)) % (2.0 * math.pi)
    lower_raw = path_to_relative_angles(lower_path, center_angle, angular_bins)
    upper_raw = path_to_relative_angles(upper_path, center_angle, angular_bins)
    lower_rel = np.minimum(lower_raw, upper_raw)
    upper_rel = np.maximum(lower_raw, upper_raw)

    voxel_rel = (theta - center_angle + math.pi) % (2.0 * math.pi) - math.pi
    low = lower_rel[radial_bin]
    high = upper_rel[radial_bin]

    if edge_softness_degrees <= 0.0:
        selector_2d = ((voxel_rel >= low) & (voxel_rel <= high)).astype(np.float32)
    else:
        softness = math.radians(edge_softness_degrees)
        distance_inside = np.minimum(voxel_rel - low, high - voxel_rel)
        selector_2d = np.clip(distance_inside / softness, 0.0, 1.0).astype(np.float32)

    selector_2d = np.where(radius <= axis_radius_pixels, 1.0, selector_2d)
    return selector_2d.astype(np.float32), center_angle, lower_rel, upper_rel


def main():
    parser = argparse.ArgumentParser(
        description="Cut one least-damaged Cn sector from a mask around the Z axis."
    )
    parser.add_argument("input_mask", type=Path)
    parser.add_argument("output_mask", type=Path)
    parser.add_argument("--symmetry", "--order", dest="order", required=True, type=parse_order)
    parser.add_argument("--angular-bins", type=int, default=720)
    parser.add_argument("--radial-bins", type=int, default=256)
    parser.add_argument("--search-window-deg", type=float, default=12.0)
    parser.add_argument("--smoothness", type=float, default=0.08)
    parser.add_argument("--radial-sigma", type=float, default=1.5)
    parser.add_argument("--angular-sigma", type=float, default=2.0)
    parser.add_argument("--edge-softness-deg", type=float, default=0.0)
    parser.add_argument("--axis-radius-pixels", type=float, default=1.5)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON metadata path. Defaults to OUTPUT_STEM.json",
    )
    args = parser.parse_args()

    if args.angular_bins < args.order * 4:
        raise SystemExit("--angular-bins must be at least four times the cyclic order")
    if args.radial_bins < 8:
        raise SystemExit("--radial-bins must be at least 8")

    volume, voxel_size, origin, starts = load_mrc(args.input_mask)
    radius, theta, radial_bin, angular_bin, max_radius = polar_maps(
        volume.shape, args.radial_bins, args.angular_bins
    )
    cost = build_cost_map(volume, radial_bin, angular_bin, args.radial_bins, args.angular_bins)
    cost = smooth_cost(cost, args.radial_sigma, args.angular_sigma)

    min_radial_bin = int(np.ceil(args.axis_radius_pixels / max_radius * (args.radial_bins - 1)))
    min_radial_bin = max(0, min(args.radial_bins - 1, min_radial_bin))
    lower_nominal, upper_nominal, nominal_cost = choose_nominal_boundary(
        cost, args.order, min_radial_bin
    )

    search_bins = int(round(args.search_window_deg / 360.0 * args.angular_bins))
    search_bins = max(1, search_bins)
    lower_path = follow_valley(cost, lower_nominal, search_bins, args.smoothness)
    upper_path = follow_valley(cost, upper_nominal, search_bins, args.smoothness)
    selector_2d, center_angle, lower_rel, upper_rel = make_sector_selector(
        theta,
        radial_bin,
        lower_nominal,
        lower_path,
        upper_path,
        args.order,
        args.angular_bins,
        args.edge_softness_deg,
        args.axis_radius_pixels,
        radius,
    )

    output = volume * selector_2d[np.newaxis, :, :]
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    write_mrc(args.output_mask, output, voxel_size, origin, starts)

    lower_angle = (lower_nominal + 0.5) / args.angular_bins * 360.0
    upper_angle = (upper_nominal + 0.5) / args.angular_bins * 360.0
    metadata_path = args.metadata or args.output_mask.with_suffix(".json")
    input_sum = float(np.sum(volume))
    output_sum = float(np.sum(output))
    output_fraction = output_sum / input_sum if input_sum != 0.0 else None
    metadata = {
        "input_mask": str(args.input_mask),
        "output_mask": str(args.output_mask),
        "symmetry": f"C{args.order}",
        "axis": "z",
        "angular_bins": args.angular_bins,
        "radial_bins": args.radial_bins,
        "search_window_degrees": args.search_window_deg,
        "smoothness": args.smoothness,
        "radial_sigma": args.radial_sigma,
        "angular_sigma": args.angular_sigma,
        "edge_softness_degrees": args.edge_softness_deg,
        "axis_radius_pixels": args.axis_radius_pixels,
        "nominal_lower_cut_degrees": lower_angle,
        "nominal_upper_cut_degrees": upper_angle,
        "sector_center_degrees": math.degrees(center_angle) % 360.0,
        "nominal_cut_cost": nominal_cost,
        "input_sum": input_sum,
        "output_sum": output_sum,
        "output_fraction_of_input_sum": output_fraction,
        "lower_boundary_degrees_by_radial_bin": (
            np.degrees(lower_rel + center_angle) % 360.0
        ).tolist(),
        "upper_boundary_degrees_by_radial_bin": (
            np.degrees(upper_rel + center_angle) % 360.0
        ).tolist(),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote sector mask: {args.output_mask}")
    print(f"Wrote metadata: {metadata_path}")
    print(
        "Chosen cuts: "
        f"{lower_angle:.3f} deg and {upper_angle:.3f} deg "
        f"(center {math.degrees(center_angle) % 360.0:.3f} deg)"
    )
    if output_fraction is None:
        print("Output/Input mask sum: undefined because input mask sum is zero")
    else:
        print(f"Output/Input mask sum: {output_fraction:.6f}")


if __name__ == "__main__":
    main()
