import json
import os
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".mplconfig"))
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import volume_viewer_pyvista_chain_palettes_gif as chain_viewer
import volume_viewer_pyvista_ken10_experimental_gif as ken10_viewer


def render_preview_image(
    mesh,
    camera_position,
    background="white",
    base_boxsize=512,
    smooth_shading=False,
    lighting_kwargs=None,
    shadows=False,
    overlay_callback=None,
):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is not installed in this Python environment. "
            "Install it with `pip install pyvista` and rerun this script."
        ) from exc

    lighting_kwargs = lighting_kwargs or {
        "lighting": True,
        "ambient": 0.30,
        "diffuse": 0.65,
        "specular": 0.12,
        "specular_power": 10.0,
    }

    preview_dir = REPO_DIR / ".preview_cache"
    preview_dir.mkdir(exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix="isecc-preview-",
        dir=preview_dir,
        delete=False,
    )
    handle.close()

    plotter = pv.Plotter(off_screen=True, window_size=(base_boxsize, base_boxsize))
    try:
        plotter.set_background(background)
        if shadows and lighting_kwargs.get("lighting"):
            try:
                plotter.enable_shadows()
            except Exception:
                pass
        plotter.add_mesh(
            mesh,
            scalars="face_rgb",
            rgb=True,
            show_edges=False,
            smooth_shading=smooth_shading,
            **lighting_kwargs,
        )
        if overlay_callback is not None:
            overlay_callback(plotter)
        plotter.camera_position = camera_position
        plotter.camera.parallel_projection = False
        plotter.reset_camera_clipping_range()
        plotter.screenshot(handle.name)
    finally:
        plotter.close()

    return handle.name


def build_chain_preview(config):
    module = chain_viewer
    volume, header_angpix, origin, starts = module.load_volume(config["input_file"])
    angpix = module.resolve_angpix(config["angpix"], header_angpix)
    world_origin = module.resolve_world_origin(origin, starts, angpix)

    level = config["level"]
    step_size = config["step_size"]
    if config["fast"]:
        step_size = max(step_size, 4)
        if level is None:
            level = round((float(volume.max()) / 6.0) + 0.004, 4)

    chain_groups = module.parse_chain_palette_specs(config["chain_palette_specs"])
    symmetry_chain_ids = module.parse_symmetry_copy_specs(config["symmetry_copy_specs"])
    verts_centered, faces, facecolors = module.extract_surface(
        ndimage=volume,
        angpix=angpix,
        world_origin=world_origin,
        cif_file=config["cif_file"],
        chain_groups=chain_groups,
        symmetry_chain_ids=symmetry_chain_ids,
        distance_cutoff=config["distance_cutoff"],
        hide_dust=config["hide_dust"],
        dust_volume_cutoff=config["dust_volume_cutoff"],
        level=level,
        step_size=step_size,
        cmap_name=config["cmap"],
    )
    verts_centered = module.apply_symmetry_operation(verts_centered, config["symop"])
    mesh = module.make_pyvista_mesh(verts_centered, faces, facecolors)
    distance = module.camera_distance_for_mesh(verts_centered)
    camera_position = module.camera_position_from_angles(
        config["elev"],
        config["azim"],
        distance,
    )
    return render_preview_image(
        mesh=mesh,
        camera_position=camera_position,
        background=config["background"],
        base_boxsize=512,
    )


def build_ken10_preview(config):
    module = ken10_viewer
    volume, header_angpix, origin, starts = module.load_volume(config["input_file"])
    angpix = module.resolve_angpix(config["angpix"], header_angpix)
    world_origin = module.resolve_world_origin(origin, starts, angpix)

    level = config["level"]
    step_size = config["step_size"]
    if config["fast"]:
        step_size = max(step_size, 4)
        if level is None:
            level = round((float(volume.max()) / 6.0) + 0.004, 4)

    component_specs = [
        {
            "label": "Major head",
            "palette": module.MAJOR_HEAD_PALETTE,
            "cif_file": config["major_head_cif"],
        },
        {
            "label": "Minor head 1",
            "palette": module.MINOR_HEAD1_PALETTE,
            "cif_file": config["minor_head1_cif"],
        },
        {
            "label": "Minor head 2",
            "palette": module.MINOR_HEAD2_PALETTE,
            "cif_file": config["minor_head2_cif"],
        },
    ]
    verts_centered, faces, facecolors = module.extract_surface(
        ndimage=volume,
        angpix=angpix,
        world_origin=world_origin,
        component_specs=component_specs,
        distance_cutoff=config["distance_cutoff"],
        hide_dust=config["hide_dust"],
        dust_volume_cutoff=config["dust_volume_cutoff"],
        level=level,
        step_size=step_size,
        cmap_name=config["cmap"],
    )
    verts_centered = module.apply_symmetry_operation(verts_centered, config["symop"])
    mesh = module.make_pyvista_mesh(verts_centered, faces, facecolors)
    distance = module.camera_distance_for_mesh(verts_centered)
    camera_position = module.camera_position_from_angles(
        config["elev"],
        config["azim"],
        distance,
    )
    return render_preview_image(
        mesh=mesh,
        camera_position=camera_position,
        background=config["background"],
        base_boxsize=512,
        smooth_shading=config.get("smooth_shading", False),
        lighting_kwargs=module.lighting_kwargs_for_preset(
            config.get("lighting_preset", "balanced")
        ),
        shadows=config.get("shadows", False),
        overlay_callback=lambda plotter: module.add_experimental_overlays(
            plotter=plotter,
            mesh=mesh,
            outline=config.get("outline", False),
            outline_color=config.get("outline_color", "black"),
            outline_width=config.get("outline_width", 1.5),
            silhouette=config.get("silhouette", False),
            silhouette_color=config.get("silhouette_color", "black"),
            silhouette_width=config.get("silhouette_width"),
        ),
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: volume_viewer_preview_snapshot.py MODE CONFIG_JSON")

    mode = sys.argv[1]
    config_path = Path(sys.argv[2])
    config = json.loads(config_path.read_text())

    if mode == "chain":
        image_path = build_chain_preview(config)
    elif mode == "ken10":
        image_path = build_ken10_preview(config)
    else:
        raise SystemExit(f"Unknown preview mode: {mode}")

    print(f"PREVIEW_IMAGE={image_path}")


if __name__ == "__main__":
    main()
