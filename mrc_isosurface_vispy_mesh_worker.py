import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
from skimage import measure

from isecc import iseccFFT_v3


def emit(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def load_volume(mrc_path):
    with mrcfile.open(mrc_path) as mrc:
        volume = np.array(mrc.data, copy=True)
        voxel_size = getattr(mrc, "voxel_size", None)
        angstrom_per_pixel = float(voxel_size.x) if voxel_size and voxel_size.x else 1.0
    #volume = np.flipud(volume)
    #volume = iseccFFT_v3.swapAxes_ndimage(volume)
    return volume, angstrom_per_pixel


def centered_vertices(verts, shape):
    center = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    return verts - center


def array_points_to_xyz(points):
    xyz_points = np.empty_like(points)
    xyz_points[..., 0] = points[..., 2]
    xyz_points[..., 1] = points[..., 1]
    xyz_points[..., 2] = points[..., 0]
    return xyz_points


def downsample_volume_to_target(volume, max_display_size):
    max_dim = int(max(volume.shape))
    if max_dim <= max_display_size:
        return volume, {"factor": 1, "shape": volume.shape}

    factor = int(np.ceil(max_dim / float(max_display_size)))
    factor = max(1, factor)
    downsampled = volume[::factor, ::factor, ::factor]
    return downsampled, {"factor": factor, "shape": downsampled.shape}


def radial_vertex_values(verts):
    return np.linalg.norm(verts, axis=1)


def build_mesh(cached_path, cached_volume, cached_angstrom_per_pixel, command):
    input_mrc = Path(command["input_mrc"])
    threshold = float(command["threshold"])
    step_size = int(command["step_size"])
    display_size = int(command["display_size"])
    output_npz = Path(command["output_npz"])

    if cached_volume is None or cached_path != input_mrc:
        emit({"type": "log", "job_id": command["job_id"], "message": f"Loading volume {input_mrc.name}"})
        cached_volume, cached_angstrom_per_pixel = load_volume(input_mrc)
        cached_path = input_mrc

    working_volume, downsample_info = downsample_volume_to_target(cached_volume, display_size)
    verts, faces, _, _ = measure.marching_cubes(
        working_volume,
        level=threshold,
        step_size=step_size,
    )
    effective_angstrom_per_voxel = float(cached_angstrom_per_pixel) * float(downsample_info["factor"])
    verts = centered_vertices(verts, working_volume.shape)
    verts = array_points_to_xyz(verts)
    verts = (verts * effective_angstrom_per_voxel).astype(np.float32)
    faces = faces.astype(np.uint32)
    vertex_values = radial_vertex_values(verts).astype(np.float32)

    np.savez_compressed(
        output_npz,
        verts=verts,
        faces=faces,
        vertex_values=vertex_values,
        threshold=np.array([threshold], dtype=np.float32),
        angstrom_per_pixel=np.array([cached_angstrom_per_pixel], dtype=np.float32),
        downsample_factor=np.array([downsample_info["factor"]], dtype=np.int32),
        downsample_shape=np.array(downsample_info["shape"], dtype=np.int32),
    )
    emit(
        {
            "type": "result",
            "job_id": command["job_id"],
            "output_npz": str(output_npz),
            "threshold": threshold,
            "downsample_factor": int(downsample_info["factor"]),
            "downsample_shape": list(downsample_info["shape"]),
        }
    )
    return cached_path, cached_volume, cached_angstrom_per_pixel


def main():
    cached_path = None
    cached_volume = None
    cached_angstrom_per_pixel = None
    emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "message": f"Invalid JSON command: {exc}"})
            continue

        cmd = command.get("cmd")
        if cmd == "shutdown":
            emit({"type": "shutdown"})
            return
        if cmd == "clear_cache":
            cached_path = None
            cached_volume = None
            cached_angstrom_per_pixel = None
            emit({"type": "cleared"})
            continue
        if cmd != "build":
            emit({"type": "error", "message": f"Unknown command: {cmd}"})
            continue

        try:
            cached_path, cached_volume, cached_angstrom_per_pixel = build_mesh(
                cached_path,
                cached_volume,
                cached_angstrom_per_pixel,
                command,
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "job_id": command.get("job_id"),
                    "message": str(exc),
                }
            )


if __name__ == "__main__":
    main()
