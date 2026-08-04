#!/usr/bin/env python3

"""Alternative native-I2 fivefold expansion matched to the validated legacy result.

This is intentionally narrow: it is a merge candidate for I2 fivefold local
reconstruction only.  It uses native I2 symmetry operators for the expansion,
but maps them into the same order/orientation convention as the validated
`--legacy-i2-i1-frame` output from ISECC_csparc_symmetryexpand_unique.py.
"""

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


def load_unique_module():
    module_path = Path(__file__).resolve().parent / "ISECC_csparc_symmetryexpand_unique.py"
    spec = importlib.util.spec_from_file_location("isecc_unique", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unique = load_unique_module()


# Same logical order as the validated weird-but-correct legacy-I2 output.
LEGACY_I1_INVERSE_FIVEFOLD_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 31],
    dtype=np.int32,
)

# Native I2 operators whose inverse-rotated fivefold vectors match the legacy
# local vectors above after the old +90 deg Z I2->I1 frame conversion.
SENSIBLE_I2_FIVEFOLD_INDICES = np.array(
    [0, 55, 1, 24, 32, 11, 30, 37, 9, 8, 6, 29],
    dtype=np.int32,
)


def i2_sensible_output_path_for(input_path, extra_suffix=""):
    input_path = Path(input_path)
    return input_path.with_name(
        f"{input_path.stem}_I2_sensibleNative_symmetryexpand_unique_fivefold_toZ{extra_suffix}.cs"
    )


def sensible_i2_fivefold_particles(
    input_path,
    output_path=None,
    subparticle_distance_angstrom=None,
    passthrough_path=None,
):
    if subparticle_distance_angstrom is None:
        raise ValueError("--subparticle-distance is required.")

    input_path = unique.resolve_input_path(input_path)
    cs = np.load(input_path)
    cs = unique.merge_passthrough_fields(cs, passthrough_path)
    if "alignments3D/pose" not in cs.dtype.fields:
        raise ValueError("Input .cs is missing required field 'alignments3D/pose'.")

    i1_ops = unique.symops.getSymOps("I1")
    i2_ops = unique.symops.getSymOps("I2")
    z90 = Quaternion(axis=(0.0, 0.0, 1.0), degrees=90)
    i1_to_z = unique.csparc_roi_quat_to_z("I1", "fivefold")

    i2_vector = np.asarray(unique.IDEAL_VECTORS["I2"]["fivefold"], dtype=np.float64)
    i2_vector = i2_vector / np.linalg.norm(i2_vector)
    subparticle_vector_angstrom = i2_vector * float(subparticle_distance_angstrom)
    unique.assert_no_duplicate_subparticles_for_sampled_particles(
        cs,
        i2_ops,
        SENSIBLE_I2_FIVEFOLD_INDICES,
        i2_vector,
        "fivefold",
        "I2",
    )

    expanded = np.empty(cs.size * SENSIBLE_I2_FIVEFOLD_INDICES.size, dtype=cs.dtype)
    rotated_vectors_angstrom = np.empty((expanded.size, 3), dtype=np.float32)

    write_start = 0
    for legacy_index, native_index in zip(LEGACY_I1_INVERSE_FIVEFOLD_INDICES, SENSIBLE_I2_FIVEFOLD_INDICES):
        write_stop = write_start + cs.size
        expanded[write_start:write_stop] = cs

        legacy_left = i1_to_z * Quaternion(i1_ops[int(legacy_index)]) * z90
        native_sym = Quaternion(i2_ops[int(native_index)])
        align_quat = legacy_left * native_sym.inverse

        for source_index in range(cs.size):
            source_pose = unique.aa2quat(cs[source_index]["alignments3D/pose"])
            expanded_pose = native_sym * source_pose
            rotated_vectors_angstrom[write_start + source_index] = expanded_pose.inverse.rotate(
                subparticle_vector_angstrom
            )
            expanded[write_start + source_index]["alignments3D/pose"] = unique.quat2aa(
                align_quat * expanded_pose
            )

        write_start = write_stop

    if "uid" in expanded.dtype.fields:
        expanded["uid"] = np.random.randint(
            1,
            9223372036854775000,
            size=expanded.size,
            dtype=np.uint64,
        )

    if output_path is None:
        output_path = i2_sensible_output_path_for(input_path, extra_suffix="_subparticle_shiftMinus")

    shifted = unique.apply_subparticle_offsets(expanded, rotated_vectors_angstrom, shift_sign=-1.0)
    unique.save_cs(output_path, shifted)
    return Path(output_path), cs.size, shifted.size


def compare_cs(candidate_path, reference_path):
    candidate = np.load(candidate_path)
    reference = np.load(reference_path)
    if candidate.shape != reference.shape:
        return [f"shape mismatch: candidate {candidate.shape}, reference {reference.shape}"]

    messages = []
    for name in reference.dtype.names:
        if name == "uid":
            continue
        if name not in candidate.dtype.names:
            messages.append(f"{name}: missing from candidate")
            continue
        if np.array_equal(candidate[name], reference[name]):
            continue
        if np.issubdtype(reference[name].dtype, np.number):
            max_diff = float(np.max(np.abs(candidate[name] - reference[name])))
            messages.append(f"{name}: max abs diff {max_diff:.6g}")
        else:
            messages.append(f"{name}: values differ")
    return messages


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Native-I2 fivefold expansion matched to the validated legacy I2->I1-frame result."
    )
    parser.add_argument("input", help="path to refined CryoSPARC particle.cs file")
    parser.add_argument("--output", "-o", default=None, help="output .cs path")
    parser.add_argument(
        "--passthrough",
        default=None,
        help="optional same-length .cs file; missing fields such as location/* will be merged before expansion",
    )
    parser.add_argument("--subparticle-distance", "--subparticle-distance-A", type=float, required=True)
    parser.add_argument("--compare-reference", default=None, help="optional reference .cs to compare after writing")
    args = parser.parse_args(argv)

    output_path, input_count, output_count = sensible_i2_fivefold_particles(
        args.input,
        output_path=args.output,
        subparticle_distance_angstrom=args.subparticle_distance,
        passthrough_path=args.passthrough,
    )
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    unique.write_command_log(output_path, command)

    print(f"Input particles:  {input_count}")
    print(f"Native I2 fivefold operators: {SENSIBLE_I2_FIVEFOLD_INDICES.tolist()}")
    print(f"Output particles: {output_count}")
    print(f"Subparticle distance: {args.subparticle_distance:.3f} A")
    print(f"Output subparticles: {output_path}")

    if args.compare_reference is not None:
        messages = compare_cs(output_path, args.compare_reference)
        if messages:
            print("Reference comparison differences excluding uid:")
            for message in messages:
                print(f"  {message}")
        else:
            print("Reference comparison: exact match excluding uid")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
