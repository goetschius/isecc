#!/usr/bin/env python3
"""3DVA ClusterMatch: rotate 3DVA clusters into a shared pose frame."""

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


def aa2quat(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(axis_angle))
    if theta == 0.0:
        return Quaternion()
    return Quaternion(axis=axis_angle / theta, angle=theta)


def quat2aa(quat):
    quat = Quaternion(quat)
    return np.asarray(quat.axis, dtype=np.float32) * np.float32(quat.radians)


def resolve_input_path(input_path):
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        return input_path

    match = re.fullmatch(r"(J\d+)_particles_(\d+)_particles\.cs", input_path.name)
    if match:
        job_name, iteration = match.groups()
        cryosparc_iteration_path = input_path.with_name(f"{job_name}_{iteration}_particles.cs")
        if cryosparc_iteration_path.is_file():
            return cryosparc_iteration_path

    candidates = []
    if input_path.parent.is_dir():
        candidates = sorted(input_path.parent.glob("*particles*.cs"))[:10]
    detail = f"Input .cs file not found: {input_path}"
    if candidates:
        candidate_text = "\n  ".join(str(candidate) for candidate in candidates)
        detail = f"{detail}\nNearby particle .cs files:\n  {candidate_text}"
    raise FileNotFoundError(detail)


def save_cs(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    npy_path = path.with_name(path.name + ".npy")
    if npy_path.exists():
        os.replace(npy_path, path)


def command_log_path_for(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_command.txt")


def write_command_log(output_path, command):
    command_log_path_for(output_path).write_text(f"{command}\n", encoding="utf-8")


def default_output_path(input_path, degrees, mode):
    input_path = Path(input_path)
    label = f"{degrees:g}".replace("-", "minus").replace(".", "p")
    return input_path.with_name(f"{input_path.stem}_3DVAClusterMatch_rotZ_{label}deg_{mode}.cs")


def merge_passthrough_fields(particles, passthrough):
    if passthrough is None:
        return particles

    extra_names = [name for name in passthrough.dtype.names if name not in particles.dtype.names]
    if not extra_names:
        return particles

    merged_dtype = np.dtype(
        particles.dtype.descr + [(name, passthrough.dtype[name]) for name in extra_names]
    )
    merged = np.empty(particles.shape, dtype=merged_dtype)
    for name in particles.dtype.names:
        merged[name] = particles[name]

    if passthrough.shape == particles.shape:
        passthrough_rows = passthrough
    else:
        if "uid" not in particles.dtype.names or "uid" not in passthrough.dtype.names:
            raise ValueError(
                f"Passthrough shape {passthrough.shape} does not match particle shape {particles.shape}, "
                "and uid fields are not available for matching."
            )
        passthrough_by_uid = {int(uid): index for index, uid in enumerate(passthrough["uid"])}
        missing_uids = [
            int(uid) for uid in particles["uid"] if int(uid) not in passthrough_by_uid
        ]
        if missing_uids:
            raise ValueError(
                f"Passthrough is missing {len(missing_uids)} particle uid(s); "
                f"first missing uid: {missing_uids[0]}"
            )
        passthrough_rows = passthrough[
            [passthrough_by_uid[int(uid)] for uid in particles["uid"]]
        ]

    for name in extra_names:
        merged[name] = passthrough_rows[name]
    return merged


def rotate_array_about_z(particles, degrees, mode):
    if "alignments3D/pose" not in particles.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    output = np.array(particles, copy=True)
    if np.isclose(float(degrees) % 360.0, 0.0):
        return output

    z_quat = Quaternion(axis=(0.0, 0.0, 1.0), degrees=float(degrees))
    if mode == "global-inverse":
        z_quat = z_quat.inverse

    for index in range(output.size):
        source_pose = aa2quat(output[index]["alignments3D/pose"])
        if mode in {"global", "global-inverse"}:
            rotated_pose = z_quat * source_pose
        elif mode == "local":
            rotated_pose = source_pose * z_quat
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        output[index]["alignments3D/pose"] = quat2aa(rotated_pose)
    return output


def rotate_particles_about_z(input_path, output_path, degrees, mode, passthrough=None):
    input_path = resolve_input_path(input_path)
    particles = np.load(input_path)
    particles = merge_passthrough_fields(particles, passthrough)
    output = rotate_array_about_z(particles, degrees, mode)

    save_cs(output_path, output)
    return input_path, Path(output_path), particles.size


def rotate_and_combine(input_rotations, output_path, mode, passthrough=None):
    rotated_arrays = []
    records = []
    for input_value, degrees_value in input_rotations:
        input_path = resolve_input_path(input_value)
        degrees = float(degrees_value)
        particles = np.load(input_path)
        particles = merge_passthrough_fields(particles, passthrough)
        rotated = rotate_array_about_z(particles, degrees, mode)
        rotated_arrays.append(rotated)
        records.append((input_path, degrees, rotated.size))

    if not rotated_arrays:
        raise ValueError("No input rotations were provided.")

    combined = np.concatenate(rotated_arrays)
    save_cs(output_path, combined)
    return Path(output_path), records, combined.size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "3DVA ClusterMatch: rotate one or more 3DVA particle clusters into "
            "a shared pose frame, without changing particle translations or defocus."
        )
    )
    parser.add_argument("input", nargs="?", help="path to CryoSPARC particles.cs file")
    parser.add_argument("--degrees", "--angle", type=float)
    parser.add_argument("--output", "-o", default=None, help="output .cs path")
    parser.add_argument(
        "--passthrough",
        default=None,
        help=(
            "optional .cs passthrough file. Missing fields are merged by row when same-length, "
            "or by uid when the input is a subset of the passthrough."
        ),
    )
    parser.add_argument(
        "--input-rotation",
        nargs=2,
        action="append",
        metavar=("INPUT", "DEGREES"),
        help=(
            "input .cs file and Z rotation in degrees. Repeat to rotate multiple inputs "
            "and concatenate them into one output."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("global", "global-inverse", "local"),
        default="global",
        help=(
            "global: z_rotation * pose, matching recent CryoSPARC convention tests; "
            "global-inverse: inverse of global; local: pose * z_rotation."
        ),
    )
    args = parser.parse_args(argv)

    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    passthrough = np.load(resolve_input_path(args.passthrough)) if args.passthrough else None

    if args.input_rotation:
        if args.input is not None or args.degrees is not None:
            parser.error("Use either positional INPUT with --degrees, or repeated --input-rotation pairs.")
        first_input = resolve_input_path(args.input_rotation[0][0])
        output_path = Path(args.output) if args.output else first_input.with_name(
            f"{first_input.stem}_3DVAClusterMatch_rotZ_combined_{args.mode}.cs"
        )
        output_path, records, output_count = rotate_and_combine(
            args.input_rotation,
            output_path,
            args.mode,
            passthrough=passthrough,
        )
        write_command_log(output_path, command)

        print("Pathway:     3DVA ClusterMatch")
        print(f"Output:      {output_path}")
        print(f"Output particles: {output_count}")
        print(f"Mode:        {args.mode}")
        if args.passthrough:
            print(f"Passthrough: {resolve_input_path(args.passthrough)}")
        for input_path, degrees, count in records:
            print(f"Input:       {input_path}  rotation={degrees:.6f} deg  particles={count}")
        print(f"Command log: {command_log_path_for(output_path)}")
        return

    if args.input is None:
        parser.error("INPUT is required unless --input-rotation is used.")
    if args.degrees is None:
        parser.error("--degrees is required unless --input-rotation is used.")

    input_path = resolve_input_path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(
        input_path, args.degrees, args.mode
    )
    input_path, output_path, count = rotate_particles_about_z(
        input_path,
        output_path,
        args.degrees,
        args.mode,
        passthrough=passthrough,
    )
    write_command_log(output_path, command)

    print("Pathway:     3DVA ClusterMatch")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Particles:   {count}")
    print(f"Z rotation:  {args.degrees:.6f} degrees")
    print(f"Mode:        {args.mode}")
    if args.passthrough:
        print(f"Passthrough: {resolve_input_path(args.passthrough)}")
    print(f"Command log: {command_log_path_for(output_path)}")


if __name__ == "__main__":
    main()
