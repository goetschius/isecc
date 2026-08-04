#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import importlib.util
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


def load_symops_module():
    module_path = Path(__file__).resolve().parent / "isecc" / "symops.py"
    spec = importlib.util.spec_from_file_location("isecc_symops", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


symops = load_symops_module()


IDEAL_VECTORS = {
    "I1": {
        "fivefold": np.array((0.000, 0.618, 1.000), dtype=np.float32),
        "threefold": np.array((0.382, 0.000, 1.000), dtype=np.float32),
        "twofold": np.array((0.000, 0.000, 1.000), dtype=np.float32),
    },
    "I2": {
        "fivefold": np.array((0.618, 0.000, 1.000), dtype=np.float32),
        "threefold": np.array((0.000, 0.382, 1.000), dtype=np.float32),
        "twofold": np.array((0.000, 0.000, 1.000), dtype=np.float32),
    },
}


def aa2quat(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(axis_angle))
    if theta == 0.0:
        return Quaternion()
    return Quaternion(axis=axis_angle / theta, angle=theta)


def quat2aa(quat):
    quat = Quaternion(quat)
    return np.asarray(quat.axis, dtype=np.float32) * np.float32(quat.radians)


def quat_between_vectors(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm == 0.0 or target_norm == 0.0:
        return Quaternion()

    source = source / source_norm
    target = target / target_norm
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return Quaternion()
    if np.isclose(dot, -1.0):
        trial_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if np.isclose(abs(float(np.dot(source, trial_axis))), 1.0):
            trial_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        axis = np.cross(source, trial_axis)
        axis = axis / np.linalg.norm(axis)
        return Quaternion(axis=axis, radians=np.pi)

    axis = np.cross(source, target)
    axis = axis / np.linalg.norm(axis)
    return Quaternion(axis=axis, radians=np.arccos(dot))


def vector_frame(vector):
    z_axis = np.asarray(vector, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    reference = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(z_axis, reference))) > 0.95:
        reference = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    x_axis = reference - (np.dot(reference, z_axis) * z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


def quat_between_frames(source_axes, target_axes):
    source_basis = np.column_stack(source_axes)
    target_basis = np.column_stack(target_axes)
    matrix = target_basis @ source_basis.T
    return Quaternion(matrix=matrix)


def csparc_roi_quat_to_z(symmetry, vertex):
    vector = np.asarray(IDEAL_VECTORS[symmetry.upper()][vertex.lower()], dtype=np.float64)
    if vertex == "fivefold":
        if symmetry.upper() == "I1":
            angle = np.arctan(np.true_divide(vector[1], vector[2]))
            return Quaternion(axis=(1.0, 0.0, 0.0), radians=angle)
        angle = np.arctan(np.true_divide(vector[0], vector[2]))
        return Quaternion(axis=(0.0, -1.0, 0.0), radians=angle)
    if vertex == "threefold":
        if symmetry.upper() == "I1":
            angle = np.arctan(np.true_divide(vector[0], vector[2]))
            return Quaternion(axis=(0.0, -1.0, 0.0), radians=angle)
        angle = np.arctan(np.true_divide(vector[1], vector[2]))
        return Quaternion(axis=(1.0, 0.0, 0.0), radians=angle)
    return Quaternion()


def unique_vertex_symop_indices(symmetry, vertex, decimals=3):
    symmetry = symmetry.upper()
    vertex = vertex.lower()
    quaternions = symops.getSymOps(symmetry)
    ideal_vector = IDEAL_VECTORS[symmetry][vertex]
    seen = set()
    unique_indices = []

    for index, sym_quat in enumerate(quaternions):
        rotated = Quaternion(sym_quat).inverse.rotate(ideal_vector)
        key = tuple(np.around(rotated, decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        unique_indices.append(index)

    return np.asarray(unique_indices, dtype=np.int32)


def inverse_unique_vertex_symop_indices(symmetry, vertex, decimals=3):
    symmetry = symmetry.upper()
    vertex = vertex.lower()
    quaternions = symops.getSymOps(symmetry)
    ideal_vector = IDEAL_VECTORS[symmetry][vertex]
    seen = set()
    unique_indices = []

    for index, sym_quat in enumerate(quaternions):
        rotated = Quaternion(sym_quat).inverse.rotate(ideal_vector)
        key = tuple(np.around(rotated, decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        unique_indices.append(index)

    return np.asarray(unique_indices, dtype=np.int32)


@dataclass(frozen=True)
class UniqueExpansionStrategy:
    symmetry: str
    vertex: str
    operator_symmetry: str
    ideal_vector_symmetry: str
    indices: np.ndarray
    align_quats_by_index: dict | None = None


def build_unique_expansion_strategy(symmetry, vertex):
    symmetry = symmetry.upper()
    vertex = vertex.lower()

    if symmetry == "I2" and vertex == "fivefold":
        i1_indices = inverse_unique_vertex_symop_indices("I1", "fivefold")
        i2_indices = np.array([0, 55, 1, 24, 32, 11, 30, 37, 9, 8, 6, 29], dtype=np.int32)
        i1_ops = symops.getSymOps("I1")
        i2_ops = symops.getSymOps("I2")
        z90 = Quaternion(axis=(0.0, 0.0, 1.0), degrees=90)
        i1_to_z = csparc_roi_quat_to_z("I1", "fivefold")
        align_quats_by_index = {}
        for i1_index, i2_index in zip(i1_indices, i2_indices):
            legacy_left = i1_to_z * Quaternion(i1_ops[int(i1_index)]) * z90
            native_sym = Quaternion(i2_ops[int(i2_index)])
            align_quats_by_index[int(i2_index)] = legacy_left * native_sym.inverse
        return UniqueExpansionStrategy(
            symmetry=symmetry,
            vertex=vertex,
            operator_symmetry="I2",
            ideal_vector_symmetry="I2",
            indices=i2_indices,
            align_quats_by_index=align_quats_by_index,
        )

    return UniqueExpansionStrategy(
        symmetry=symmetry,
        vertex=vertex,
        operator_symmetry=symmetry,
        ideal_vector_symmetry=symmetry,
        indices=inverse_unique_vertex_symop_indices(symmetry, vertex),
    )


def assert_no_duplicate_subparticles_for_sampled_particles(
    cs,
    quaternions,
    selected_indices,
    ideal_vector,
    vertex,
    symmetry,
    legacy_frame_quat=None,
    tolerance=1e-3,
    sample_fraction=0.01,
):
    if cs.size == 0:
        raise ValueError("Input .cs contains no particles; cannot validate unique subparticles.")
    if "alignments3D/pose" not in cs.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    legacy_frame_quat = legacy_frame_quat or Quaternion()
    ideal_unit_vector = np.asarray(ideal_vector, dtype=np.float64)
    ideal_unit_vector = ideal_unit_vector / np.linalg.norm(ideal_unit_vector)
    sample_count = max(1, int(np.ceil(cs.size * float(sample_fraction))))
    sample_count = min(sample_count, cs.size)
    sampled_indices = np.unique(np.linspace(0, cs.size - 1, sample_count, dtype=np.int64))

    for source_index in sampled_indices:
        source_pose = aa2quat(cs[int(source_index)]["alignments3D/pose"])
        working_pose = legacy_frame_quat * source_pose

        groups = []
        for sym_index in selected_indices:
            sym_quat = Quaternion(quaternions[int(sym_index)])
            expanded_pose = sym_quat * working_pose
            vector = np.asarray(expanded_pose.inverse.rotate(ideal_unit_vector), dtype=np.float64)
            vector = vector / np.linalg.norm(vector)

            duplicate_group = None
            for group in groups:
                if np.linalg.norm(vector - group["mean"]) < tolerance:
                    duplicate_group = group
                    break
            if duplicate_group is None:
                groups.append(
                    {
                        "representative": int(sym_index),
                        "indices": [int(sym_index)],
                        "vectors": [vector],
                        "mean": vector,
                    }
                )
                continue

            duplicate_group["indices"].append(int(sym_index))
            duplicate_group["vectors"].append(vector)
            duplicate_group["mean"] = np.mean(duplicate_group["vectors"], axis=0)
            duplicate_group["mean"] = duplicate_group["mean"] / np.linalg.norm(duplicate_group["mean"])

        duplicate_groups = [group for group in groups if len(group["indices"]) > 1]
        if duplicate_groups:
            details = []
            for group in duplicate_groups:
                details.append(
                    "ops "
                    + ",".join(str(index) for index in group["indices"])
                    + " -> "
                    + np.array2string(np.asarray(group["mean"]), precision=6, separator=", ")
                )
            raise ValueError(
                f"Duplicate {symmetry.upper()} {vertex.lower()} subparticles detected for sampled input particle "
                f"{int(source_index)}. Expected {len(selected_indices)} distinct subparticles, found {len(groups)}. "
                f"Checked {len(sampled_indices)} of {cs.size} particles ({100.0 * len(sampled_indices) / cs.size:.2f}%). "
                "Duplicate groups: "
                + "; ".join(details)
            )


def assert_no_duplicate_subparticles_for_first_particle(*args, **kwargs):
    kwargs["sample_fraction"] = 1.0 / max(1, getattr(args[0], "size", 1))
    return assert_no_duplicate_subparticles_for_sampled_particles(*args, **kwargs)


def normalize_align_to_z_mode(mode):
    aliases = {
        "direct": "debug-per-vertex",
        "inverse": "debug-per-vertex-inverse",
        "relion": "csparc",
        "relion-inverse": "csparc-inverse",
    }
    return aliases.get(str(mode).lower(), str(mode).lower())


def output_path_for(input_path, symmetry, vertex, align_to_z=False, align_to_z_mode="csparc", extra_suffix=""):
    input_path = Path(input_path)
    align_to_z_mode = normalize_align_to_z_mode(align_to_z_mode)
    z_suffix = ""
    if align_to_z:
        suffixes = {
            "csparc": "_toZ",
            "csparc-inverse": "_toZcsparcInv",
            "debug-per-vertex": "_toZdebugPerVertex",
            "debug-per-vertex-inverse": "_toZdebugPerVertexInv",
        }
        z_suffix = suffixes[align_to_z_mode]
    return input_path.with_name(
        f"{input_path.stem}_{symmetry.upper()}_symmetryexpand_unique_{vertex.lower()}{z_suffix}{extra_suffix}.cs"
    )


def custom_output_path_for(input_path, symmetry, align_to_z=False, align_to_z_mode="debug-per-vector", extra_suffix=""):
    input_path = Path(input_path)
    align_to_z_mode = normalize_align_to_z_mode(align_to_z_mode)
    z_suffix = ""
    if align_to_z:
        suffixes = {
            "csparc": "_toZ",
            "csparc-inverse": "_toZinv",
            "debug-per-vertex": "_toZ",
            "debug-per-vertex-inverse": "_toZinv",
            "debug-per-vector": "_toZ",
            "debug-per-vector-inverse": "_toZinv",
        }
        z_suffix = suffixes[align_to_z_mode]
    return input_path.with_name(
        f"{input_path.stem}_{symmetry.upper()}_symmetryexpand_full_customVector{z_suffix}{extra_suffix}.cs"
    )


def walk_along_z_output_path_for(input_path, extra_suffix=""):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_WalkAlongZ{extra_suffix}.cs")


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
    np.save(path, array)
    npy_path = path.with_name(path.name + ".npy")
    if npy_path.exists():
        os.replace(npy_path, path)


def command_log_path_for(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_command.txt")


def write_command_log(output_path, command):
    command_log_path_for(output_path).write_text(f"{command}\n", encoding="utf-8")


def merge_passthrough_fields(cs, passthrough_path):
    if passthrough_path is None:
        return cs
    passthrough_path = resolve_input_path(passthrough_path)
    passthrough = np.load(passthrough_path)
    extra_names = [name for name in passthrough.dtype.names if name not in cs.dtype.names]
    if not extra_names:
        return cs

    merged_dtype = np.dtype(cs.dtype.descr + [(name, passthrough.dtype[name]) for name in extra_names])
    merged = np.empty(cs.shape, dtype=merged_dtype)
    for name in cs.dtype.names:
        merged[name] = cs[name]

    if passthrough.shape == cs.shape:
        passthrough_rows = passthrough
    else:
        if "uid" not in cs.dtype.names or "uid" not in passthrough.dtype.names:
            raise ValueError(
                f"Passthrough shape {passthrough.shape} does not match particle shape {cs.shape}, "
                "and uid fields are not available for matching."
            )
        passthrough_by_uid = {int(uid): index for index, uid in enumerate(passthrough["uid"])}
        missing_uids = [int(uid) for uid in cs["uid"] if int(uid) not in passthrough_by_uid]
        if missing_uids:
            raise ValueError(
                f"Passthrough is missing {len(missing_uids)} particle uid(s); first missing uid: {missing_uids[0]}"
            )
        passthrough_rows = passthrough[[passthrough_by_uid[int(uid)] for uid in cs["uid"]]]

    for name in extra_names:
        merged[name] = passthrough_rows[name]
    return merged


def apply_subparticle_offsets(expanded, rotated_vectors_angstrom, shift_sign):
    if "alignments3D/shift" not in expanded.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/shift'.")
    if "ctf/df1_A" not in expanded.dtype.fields or "ctf/df2_A" not in expanded.dtype.fields:
        raise ValueError("Input .cs is missing required CTF defocus fields.")
    if "alignments3D/psize_A" in expanded.dtype.fields:
        pixel_size_angstrom = expanded["alignments3D/psize_A"].astype(np.float32)
    elif "blob/psize_A" in expanded.dtype.fields:
        pixel_size_angstrom = expanded["blob/psize_A"].astype(np.float32)
    else:
        raise ValueError("Input .cs is missing pixel size field needed to convert Angstrom shifts to pixels.")
    pixel_size_angstrom = np.maximum(pixel_size_angstrom, np.float32(1e-6))
    rotated_vectors_pixels = rotated_vectors_angstrom[:, :2] / pixel_size_angstrom[:, None]
    shifted = expanded.copy(order="C")
    shifted["alignments3D/shift"][:, 0] = expanded["alignments3D/shift"][:, 0] + (shift_sign * rotated_vectors_pixels[:, 0])
    shifted["alignments3D/shift"][:, 1] = expanded["alignments3D/shift"][:, 1] + (shift_sign * rotated_vectors_pixels[:, 1])
    shifted["ctf/df1_A"] = expanded["ctf/df1_A"] + rotated_vectors_angstrom[:, 2]
    shifted["ctf/df2_A"] = expanded["ctf/df2_A"] + rotated_vectors_angstrom[:, 2]
    return shifted


def apply_z_offset_without_expansion(
    input_path,
    output_path=None,
    subparticle_distance_angstrom=None,
    passthrough_path=None,
    shift_sign=-1.0,
):
    if subparticle_distance_angstrom is None:
        raise ValueError("--subparticle-distance is required with --walk-along-z.")

    input_path = resolve_input_path(input_path)
    cs = np.load(input_path)
    cs = merge_passthrough_fields(cs, passthrough_path)
    shifted = apply_z_offset_array_without_expansion(cs, subparticle_distance_angstrom, shift_sign=shift_sign)

    if output_path is None:
        output_path = walk_along_z_output_path_for(input_path)
    save_cs(output_path, shifted)
    return Path(output_path), cs.size, cs.size


def apply_z_offset_array_without_expansion(
    cs,
    subparticle_distance_angstrom,
    shift_sign=-1.0,
):
    if "alignments3D/pose" not in cs.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    z_vector_angstrom = np.array((0.0, 0.0, float(subparticle_distance_angstrom)), dtype=np.float64)
    rotated_vectors_angstrom = np.empty((cs.size, 3), dtype=np.float32)
    for source_index in range(cs.size):
        source_pose = aa2quat(cs[source_index]["alignments3D/pose"])
        rotated_vectors_angstrom[source_index] = source_pose.inverse.rotate(z_vector_angstrom)

    return apply_subparticle_offsets(cs, rotated_vectors_angstrom, shift_sign=shift_sign)


def apply_z_offset_and_combine_without_expansion(
    input_paths,
    output_path,
    subparticle_distance_angstrom=None,
    passthrough_path=None,
    shift_sign=-1.0,
):
    if subparticle_distance_angstrom is None:
        raise ValueError("--subparticle-distance is required with --walk-along-z.")
    if output_path is None:
        raise ValueError("--output is required when combining WalkAlongZ inputs.")

    shifted_arrays = []
    input_counts = []
    for input_path in input_paths:
        resolved_path = resolve_input_path(input_path)
        cs = np.load(resolved_path)
        cs = merge_passthrough_fields(cs, passthrough_path)
        shifted = apply_z_offset_array_without_expansion(
            cs,
            subparticle_distance_angstrom,
            shift_sign=shift_sign,
        )
        shifted_arrays.append(shifted)
        input_counts.append((resolved_path, cs.size))

    combined = np.concatenate(shifted_arrays)
    save_cs(output_path, combined)
    return Path(output_path), input_counts, combined.size


def symmetry_expand_full_custom_vector_particles(
    input_path,
    symmetry,
    custom_vector,
    output_path=None,
    align_to_z=True,
    align_to_z_mode="csparc",
    subparticle_distance_angstrom=None,
    passthrough_path=None,
):
    symmetry = symmetry.upper()
    align_to_z_mode = normalize_align_to_z_mode(align_to_z_mode)
    mode_aliases = {
        "csparc": "fixed-vector",
        "debug-per-vertex": "debug-per-vector",
        "debug-per-vertex-inverse": "debug-per-vector-inverse",
        "csparc-inverse": "debug-per-vector",
    }
    align_to_z_mode = mode_aliases.get(align_to_z_mode, align_to_z_mode)
    valid_align_modes = {
        "csparc-frame",
        "fixed-vector",
        "fixed-vector-inverse",
        "debug-per-vector",
        "debug-per-vector-inverse",
    }
    if align_to_z_mode not in valid_align_modes:
        raise ValueError(f"align_to_z_mode must be one of {sorted(valid_align_modes)} for custom vectors.")

    custom_vector = np.asarray(custom_vector, dtype=np.float64)
    custom_norm = float(np.linalg.norm(custom_vector))
    if custom_norm <= 0.0:
        raise ValueError("--custom-vector must have nonzero length.")
    custom_unit_vector = custom_vector / custom_norm
    custom_frame = vector_frame(custom_unit_vector)
    fixed_vector_quat_to_z = quat_between_vectors(custom_unit_vector, target=np.array((0.0, 0.0, 1.0)))
    target_frame = (
        np.array((1.0, 0.0, 0.0), dtype=np.float64),
        np.array((0.0, 1.0, 0.0), dtype=np.float64),
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
    )

    input_path = resolve_input_path(input_path)
    cs = np.load(input_path)
    cs = merge_passthrough_fields(cs, passthrough_path)
    if "alignments3D/pose" not in cs.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    quaternions = symops.getSymOps(symmetry)
    full_indices = np.arange(len(quaternions), dtype=np.int32)
    target_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    expanded = np.empty(cs.size * full_indices.size, dtype=cs.dtype)
    rotated_vectors_angstrom = None
    if subparticle_distance_angstrom is not None:
        subparticle_vector_angstrom = custom_unit_vector * float(subparticle_distance_angstrom)
        rotated_vectors_angstrom = np.empty((expanded.size, 3), dtype=np.float32)

    write_start = 0
    for sym_index in full_indices:
        write_stop = write_start + cs.size
        expanded[write_start:write_stop] = cs
        sym_quat = Quaternion(quaternions[int(sym_index)])
        align_quat = Quaternion()
        if align_to_z:
            if align_to_z_mode == "debug-per-vector":
                expanded_vector = sym_quat.rotate(custom_unit_vector)
                align_quat = quat_between_vectors(expanded_vector, target_z)
            elif align_to_z_mode == "debug-per-vector-inverse":
                align_quat = quat_between_vectors(sym_quat.rotate(target_z), custom_unit_vector)
                align_quat = align_quat.inverse
            elif align_to_z_mode == "fixed-vector":
                align_quat = fixed_vector_quat_to_z
            elif align_to_z_mode == "fixed-vector-inverse":
                align_quat = fixed_vector_quat_to_z.inverse
            elif align_to_z_mode == "csparc-frame":
                source_frame = tuple(np.asarray(sym_quat.rotate(axis), dtype=np.float64) for axis in custom_frame)
                align_quat = quat_between_frames(source_frame, target_frame)

        for source_index in range(cs.size):
            source_pose = aa2quat(cs[source_index]["alignments3D/pose"])
            expanded_pose = sym_quat * source_pose
            if rotated_vectors_angstrom is not None:
                rotated_vectors_angstrom[write_start + source_index] = expanded_pose.inverse.rotate(
                    subparticle_vector_angstrom
                )
            if align_to_z:
                expanded_pose = align_quat * expanded_pose
            expanded[write_start + source_index]["alignments3D/pose"] = quat2aa(expanded_pose)

        write_start = write_stop

    if "uid" in expanded.dtype.fields:
        expanded["uid"] = np.random.randint(
            1,
            9223372036854775000,
            size=expanded.size,
            dtype=np.uint64,
        )

    extra_suffix = "_subparticle" if subparticle_distance_angstrom is not None else ""
    if output_path is None:
        output_path = custom_output_path_for(
            input_path,
            symmetry,
            align_to_z=align_to_z,
            align_to_z_mode=align_to_z_mode,
            extra_suffix=extra_suffix,
        )

    if subparticle_distance_angstrom is not None:
        save_cs(output_path, apply_subparticle_offsets(expanded, rotated_vectors_angstrom, shift_sign=-1.0))
    else:
        save_cs(output_path, expanded)
    return Path(output_path), cs.size, expanded.size, full_indices


def symmetry_expand_unique_particles(
    input_path,
    symmetry,
    vertex,
    output_path=None,
    align_to_z=False,
    align_to_z_mode="csparc",
    subparticle_distance_angstrom=None,
    passthrough_path=None,
    write_debug_shift_plus=False,
):
    symmetry = symmetry.upper()
    vertex = vertex.lower()
    align_to_z_mode = normalize_align_to_z_mode(align_to_z_mode)
    valid_align_modes = {"csparc", "csparc-inverse", "debug-per-vertex", "debug-per-vertex-inverse"}
    if align_to_z_mode not in valid_align_modes:
        raise ValueError(f"align_to_z_mode must be one of {sorted(valid_align_modes)}.")
    input_path = resolve_input_path(input_path)
    cs = np.load(input_path)
    cs = merge_passthrough_fields(cs, passthrough_path)
    if "alignments3D/pose" not in cs.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    strategy = build_unique_expansion_strategy(symmetry, vertex)
    quaternions = symops.getSymOps(strategy.operator_symmetry)
    unique_indices = strategy.indices
    ideal_vector = IDEAL_VECTORS[strategy.ideal_vector_symmetry][vertex]
    target_z = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    csparc_quat_to_z = csparc_roi_quat_to_z(strategy.ideal_vector_symmetry, vertex)
    assert_no_duplicate_subparticles_for_sampled_particles(
        cs,
        quaternions,
        unique_indices,
        ideal_vector,
        vertex,
        symmetry,
    )
    expanded = np.empty(cs.size * unique_indices.size, dtype=cs.dtype)
    rotated_vectors_angstrom = None
    if subparticle_distance_angstrom is not None:
        ideal_unit_vector = np.asarray(ideal_vector, dtype=np.float64)
        ideal_unit_vector = ideal_unit_vector / np.linalg.norm(ideal_unit_vector)
        subparticle_vector_angstrom = ideal_unit_vector * float(subparticle_distance_angstrom)
        rotated_vectors_angstrom = np.empty((expanded.size, 3), dtype=np.float32)

    write_start = 0
    for sym_index in unique_indices:
        write_stop = write_start + cs.size
        expanded[write_start:write_stop] = cs
        sym_quat = Quaternion(quaternions[int(sym_index)])
        align_quat = Quaternion()
        if align_to_z:
            if align_to_z_mode in {"csparc", "csparc-inverse"}:
                if strategy.align_quats_by_index:
                    align_quat = strategy.align_quats_by_index.get(int(sym_index), csparc_quat_to_z)
                else:
                    align_quat = csparc_quat_to_z
            else:
                expanded_vertex = sym_quat.rotate(ideal_vector)
                align_quat = quat_between_vectors(expanded_vertex, target_z)
            if align_to_z_mode in {"debug-per-vertex-inverse", "csparc-inverse"}:
                align_quat = align_quat.inverse

        for source_index in range(cs.size):
            source_pose = aa2quat(cs[source_index]["alignments3D/pose"])
            expanded_pose = sym_quat * source_pose
            if rotated_vectors_angstrom is not None:
                rotated_vectors_angstrom[write_start + source_index] = expanded_pose.inverse.rotate(
                    subparticle_vector_angstrom
                )
            if align_to_z:
                expanded_pose = align_quat * expanded_pose
            expanded[write_start + source_index]["alignments3D/pose"] = quat2aa(expanded_pose)

        write_start = write_stop

    if "uid" in expanded.dtype.fields:
        expanded["uid"] = np.random.randint(
            1,
            9223372036854775000,
            size=expanded.size,
            dtype=np.uint64,
        )

    if subparticle_distance_angstrom is not None:
        if output_path is not None:
            subparticle_path = Path(output_path)
        else:
            subparticle_path = output_path_for(
                input_path,
                symmetry,
                vertex,
                align_to_z=align_to_z,
                align_to_z_mode=align_to_z_mode,
                extra_suffix="_subparticle",
            )
        save_cs(subparticle_path, apply_subparticle_offsets(expanded, rotated_vectors_angstrom, shift_sign=-1.0))
        if write_debug_shift_plus:
            debug_path = subparticle_path.with_name(f"{subparticle_path.stem}_DEBUG_DISFAVORED_shiftPlus.cs")
            save_cs(debug_path, apply_subparticle_offsets(expanded, rotated_vectors_angstrom, shift_sign=1.0))
            return (Path(subparticle_path), Path(debug_path)), cs.size, expanded.size, unique_indices
        return Path(subparticle_path), cs.size, expanded.size, unique_indices

    if output_path is None:
        output_path = output_path_for(
            input_path,
            symmetry,
            vertex,
            align_to_z=align_to_z,
            align_to_z_mode=align_to_z_mode,
        )
    save_cs(output_path, expanded)
    return Path(output_path), cs.size, expanded.size, unique_indices


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Symmetry expand a CryoSPARC particle.cs file, keeping one pose per unique vertex."
    )
    parser.add_argument("input", nargs="?", help="path to refined CryoSPARC particle.cs file")
    parser.add_argument("--symmetry", "--supersym", choices=("I1", "I2"), type=str.upper, default="I1")
    parser.add_argument("--vertex", "--roi", choices=("fivefold", "threefold", "twofold"), type=str.lower)
    parser.add_argument("--output", "-o", default=None, help="output .cs path")
    parser.add_argument(
        "--passthrough",
        default=None,
        help="optional same-length .cs file; missing fields such as location/* will be merged before expansion",
    )
    parser.add_argument(
        "--align-to-z",
        "--rotate-to-z",
        dest="align_to_z",
        action="store_true",
        default=None,
        help=(
            "post-rotate expanded poses toward the +Z axis. "
            "I2 fivefold uses the sensible native-I2 strategy by default."
        ),
    )
    parser.add_argument(
        "--no-align-to-z",
        dest="align_to_z",
        action="store_false",
        help="do not post-rotate expanded poses to the +Z axis",
    )
    parser.add_argument(
        "--align-to-z-mode",
        metavar="{csparc,csparc-inverse,debug-per-vertex,debug-per-vertex-inverse,...}",
        default="csparc",
        help=(
            "csparc applies the fixed CryoSPARC ROI correction for one ideal vertex. "
            "debug-per-vertex aligns every symmetry-related unique vertex to +Z, "
            "and inverse/custom-vector modes are convention checks."
        ),
    )
    parser.add_argument(
        "--subparticle-distance",
        "--subparticle-distance-A",
        type=float,
        default=None,
        help=(
            "define subparticle centers this many Angstroms along the selected idealized vertex vector. "
            "Uses the validated CryoSPARC shift convention by default."
        ),
    )
    parser.add_argument(
        "--write-debug-shift-plus",
        action="store_true",
        help=(
            "also write the opposite-sign subparticle shift file. "
            "This is debug-only and explicitly disfavored for normal use."
        ),
    )
    parser.add_argument(
        "--walk-along-z",
        "--no-symmetry-expand",
        dest="walk_along_z",
        action="store_true",
        help=(
            "preserve one output row per input row and apply only a local +Z offset. "
            "This is the WalkAlongZ pathway; --no-symmetry-expand is retained as a legacy alias. "
            "Use a negative --subparticle-distance to reverse a previous +Z subparticle shift."
        ),
    )
    parser.add_argument(
        "--combine-input",
        action="append",
        default=None,
        help=(
            "with --walk-along-z, repeat this for multiple input .cs files to "
            "apply the same Z offset and concatenate them into one --output file"
        ),
    )
    parser.add_argument(
        "--full-symmetry-expand",
        action="store_true",
        help="expand through all symmetry operators instead of keeping only unique ideal vertices",
    )
    parser.add_argument(
        "--custom-vector",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="arbitrary local vector for full symmetry expansion; normalized before use",
    )
    args = parser.parse_args(argv)
    align_to_z = args.align_to_z
    if align_to_z is None:
        align_to_z = args.custom_vector is not None
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    if args.walk_along_z:
        if align_to_z:
            parser.error("--align-to-z is not used with --walk-along-z because poses are preserved.")
        if args.write_debug_shift_plus:
            parser.error("--write-debug-shift-plus is only available during symmetry expansion.")
        if args.combine_input:
            if args.input is not None:
                parser.error("Use either positional INPUT or repeated --combine-input values, not both.")
            if args.output is None:
                parser.error("--output is required with --combine-input.")
            output_path, input_counts, output_count = apply_z_offset_and_combine_without_expansion(
                args.combine_input,
                output_path=args.output,
                subparticle_distance_angstrom=args.subparticle_distance,
                passthrough_path=args.passthrough,
            )
            write_command_log(output_path, command)
            print("Pathway: WalkAlongZ")
            print("Symmetry expanded: False")
            for input_path, input_count in input_counts:
                print(f"Input particles:  {input_count}  {input_path}")
            print(f"Output particles: {output_count}")
            print(f"Z offset distance: {args.subparticle_distance:.3f} A")
            print(f"Output: {output_path}")
            return
        if args.input is None:
            parser.error("INPUT is required unless --combine-input is used.")
        output_path, input_count, output_count = apply_z_offset_without_expansion(
            args.input,
            output_path=args.output,
            subparticle_distance_angstrom=args.subparticle_distance,
            passthrough_path=args.passthrough,
        )
        write_command_log(output_path, command)
        print("Pathway: WalkAlongZ")
        print(f"Input particles:  {input_count}")
        print("Symmetry expanded: False")
        print(f"Output particles: {output_count}")
        print(f"Z offset distance: {args.subparticle_distance:.3f} A")
        print(f"Output: {output_path}")
        return

    if args.vertex is None:
        if args.custom_vector is None:
            parser.error("--vertex/--roi is required unless --walk-along-z or --custom-vector is set.")
    if args.input is None:
        parser.error("INPUT is required unless --walk-along-z with --combine-input is used.")

    if args.custom_vector is not None:
        if not args.full_symmetry_expand:
            parser.error("--custom-vector requires --full-symmetry-expand.")
        output_path, input_count, output_count, full_indices = symmetry_expand_full_custom_vector_particles(
            args.input,
            args.symmetry,
            args.custom_vector,
            output_path=args.output,
            align_to_z=align_to_z,
            align_to_z_mode=args.align_to_z_mode,
            subparticle_distance_angstrom=args.subparticle_distance,
            passthrough_path=args.passthrough,
        )
        write_command_log(output_path, command)
        print(f"Input particles:  {input_count}")
        print(f"Full symmetry operators: {len(full_indices)}")
        print(f"Output particles: {output_count}")
        print(f"Custom vector:    {args.custom_vector[0]:.6f}, {args.custom_vector[1]:.6f}, {args.custom_vector[2]:.6f}")
        print(f"Aligned to Z:     {align_to_z}")
        if align_to_z:
            print(f"Z align mode:     {normalize_align_to_z_mode(args.align_to_z_mode)}")
        if args.subparticle_distance is not None:
            print(f"Subparticle distance: {args.subparticle_distance:.3f} A")
            print(f"Output subparticles: {output_path}")
        else:
            print(f"Output: {output_path}")
        return

    output_path, input_count, output_count, unique_indices = symmetry_expand_unique_particles(
        args.input,
        args.symmetry,
        args.vertex,
        output_path=args.output,
        align_to_z=align_to_z,
        align_to_z_mode=args.align_to_z_mode,
        subparticle_distance_angstrom=args.subparticle_distance,
        passthrough_path=args.passthrough,
        write_debug_shift_plus=args.write_debug_shift_plus,
    )
    print(f"Input particles:  {input_count}")
    print(f"Unique {args.vertex} vertices: {len(unique_indices)}")
    print(f"Output particles: {output_count}")
    print(f"Aligned to Z:     {align_to_z}")
    if align_to_z:
        print(f"Z align mode:     {normalize_align_to_z_mode(args.align_to_z_mode)}")
    if args.subparticle_distance is not None:
        print(f"Subparticle distance: {args.subparticle_distance:.3f} A")
    if args.write_debug_shift_plus:
        write_command_log(output_path[0], command)
        write_command_log(output_path[1], command)
        print(f"Output subparticles:       {output_path[0]}")
        print(f"DEBUG disfavored shiftPlus: {output_path[1]}")
    else:
        write_command_log(output_path, command)
        print(f"Output subparticles: {output_path}" if args.subparticle_distance is not None else f"Output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
