import os
import json
import sys
import tempfile
from pathlib import Path

import mrcfile
import numpy as np
from matplotlib import colormaps
from PySide6 import QtCore, QtGui, QtWidgets
from vispy.geometry import create_sphere
from vispy import scene
from vispy.visuals.transforms import MatrixTransform

from isecc import iseccFFT_v3


REPO_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".mplconfig"))

PALETTES = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "turbo",
    "rainbow",
    "rainbow_r",
    "cubehelix",
    "Spectral",
    "coolwarm",
)

COLORING_METHODS = (
    ("Radial (sphere)", "radial_sphere"),
    ("Radial (cylinder-Z)", "radial_cylinder_z"),
    ("Radial (cylinder-Y)", "radial_cylinder_y"),
    ("Radial (cylinder-X)", "radial_cylinder_x"),
)

VECTOR_LINE_WIDTH = 3.0
VECTOR_MARKER_SIZE = 5.0
SPHERE_COLOR = np.array((0.18, 0.55, 0.95, 0.22), dtype=np.float32)
SPHERE_OUTLINE_COLOR = np.array((0.05, 0.10, 0.18, 0.85), dtype=np.float32)
SPHERE_OUTLINE_GROWTH_ANGSTROM = 1.5
SECTION_SPHERE_OUTLINE_COLOR = QtGui.QColor(18, 42, 82, 220)
SECTION_SPHERE_FILL_COLOR = QtGui.QColor(46, 140, 242, 70)
SECTION_SPHERE_LINE_WIDTH = 2.0
SECTION_VECTOR_LINE_WIDTH = 2.0
SECTION_VECTOR_TIP_RADIUS = 3.0
SPHERE_ROWS = 18
SPHERE_COLS = 24
VECTOR_SPECS = (
    {
        "key": "threefold",
        "label": "Threefold",
        "color_name": "Red",
        "color": np.array((0.86, 0.23, 0.16, 1.0), dtype=np.float32),
    },
    {
        "key": "fivefold",
        "label": "Fivefold",
        "color_name": "Blue",
        "color": np.array((0.16, 0.39, 0.86, 1.0), dtype=np.float32),
    },
    {
        "key": "twofold",
        "label": "Twofold",
        "color_name": "Green",
        "color": np.array((0.12, 0.65, 0.31, 1.0), dtype=np.float32),
    },
)
VECTOR_SET_DEFINITIONS = {
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


def load_volume(mrc_path):
    with mrcfile.open(mrc_path) as mrc:
        volume = np.array(mrc.data, copy=True)
        voxel_size = getattr(mrc, "voxel_size", None)
        angstrom_per_pixel = float(voxel_size.x) if voxel_size and voxel_size.x else 1.0
    #volume = np.flipud(volume)
    #volume = iseccFFT_v3.swapAxes_ndimage(volume)
    return volume, angstrom_per_pixel


def default_threshold(volume):
    return float(np.around((np.amax(volume) / 10.0) + 0.004, decimals=4))


def centered_vertices(verts, shape):
    center = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    return verts - center


def array_shape_to_xyz(shape):
    shape = np.asarray(shape)
    return shape[[2, 1, 0]]


def display_space_points(points):
    display_points = np.empty_like(points)
    display_points[..., 0] = points[..., 0]
    display_points[..., 1] = -points[..., 2]
    display_points[..., 2] = points[..., 1]
    return display_points


def downsample_volume_to_target(volume, max_display_size):
    max_dim = int(max(volume.shape))
    if max_dim <= max_display_size:
        return volume, {"factor": 1, "shape": volume.shape}

    factor = int(np.ceil(max_dim / float(max_display_size)))
    factor = max(1, factor)
    downsampled = volume[::factor, ::factor, ::factor]
    return downsampled, {"factor": factor, "shape": downsampled.shape}


def radial_values_from_components(x_values, y_values, z_values, method):
    if method == "radial_sphere":
        return np.sqrt((x_values ** 2) + (y_values ** 2) + (z_values ** 2))
    if method == "radial_cylinder_z":
        return np.sqrt((x_values ** 2) + (y_values ** 2))
    if method == "radial_cylinder_y":
        return np.sqrt((x_values ** 2) + (z_values ** 2))
    if method == "radial_cylinder_x":
        return np.sqrt((y_values ** 2) + (z_values ** 2))
    raise ValueError(f"Unsupported coloring method: {method}")


def radial_vertex_values(verts, method="radial_sphere"):
    verts = np.asarray(verts, dtype=np.float32)
    return radial_values_from_components(verts[:, 0], verts[:, 1], verts[:, 2], method)


def palette_colors(values, palette, gradient_start=None, gradient_stop=None):
    cmap = colormaps.get_cmap(palette)
    values = np.asarray(values, dtype=np.float32)
    auto_min = float(np.min(values))
    auto_max = float(np.max(values))
    vmin = auto_min if gradient_start is None else float(gradient_start)
    vmax = auto_max if gradient_stop is None else float(gradient_stop)
    if vmax <= vmin:
        vmin = auto_min
        vmax = auto_max
    if np.isclose(vmin, vmax):
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    return cmap(normalized).astype(np.float32)


def make_qpixmap(rgb_image):
    rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
    height, width, _channels = rgb_image.shape
    qimage = QtGui.QImage(
        rgb_image.data,
        width,
        height,
        rgb_image.strides[0],
        QtGui.QImage.Format_RGB888,
    ).copy()
    return QtGui.QPixmap.fromImage(qimage)


class SliceView(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_label = QtWidgets.QLabel("No data")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(160, 160)
        self.image_label.setStyleSheet("background: rgb(245, 247, 250); border: 1px solid rgb(220, 225, 232);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)

        self._pixmap = None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._refresh_pixmap()

    def clear(self, text="No data"):
        self._pixmap = None
        self.image_label.setText(text)
        self.image_label.setPixmap(QtGui.QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")


class ClampedTurntableCamera(scene.cameras.TurntableCamera):
    def __init__(self, *args, **kwargs):
        self.zoom_limits = None
        super().__init__(*args, **kwargs)

    def set_zoom_limits(self, min_scale, max_scale, min_distance, max_distance):
        self.zoom_limits = {
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "min_distance": float(min_distance),
            "max_distance": float(max_distance),
        }
        self._apply_zoom_limits()

    def _apply_zoom_limits(self):
        if not self.zoom_limits:
            return
        self._scale_factor = min(
            max(float(self._scale_factor), self.zoom_limits["min_scale"]),
            self.zoom_limits["max_scale"],
        )
        if self._distance is not None:
            self._distance = min(
                max(float(self._distance), self.zoom_limits["min_distance"]),
                self.zoom_limits["max_distance"],
            )

    def view_changed(self):
        self._apply_zoom_limits()
        super().view_changed()


class VispyIsosurfaceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.default_elevation = 0.0
        self.default_azimuth = 0.0

        self.canvas = scene.SceneCanvas(
            keys=None,
            bgcolor=(0.96, 0.97, 0.98, 1.0),
            size=(900, 900),
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = ClampedTurntableCamera(
            fov=45.0,
            elevation=self.default_elevation,
            azimuth=self.default_azimuth,
            distance=250.0,
            up="+z",
        )
        self.object_node = scene.Node(parent=self.view.scene)
        self.mesh = scene.visuals.Mesh(
            shading=None,
            parent=self.object_node,
        )
        self.sphere_outline_mesh = scene.visuals.Mesh(
            color=SPHERE_OUTLINE_COLOR,
            shading=None,
            parent=self.object_node,
        )
        self.sphere_outline_mesh.set_gl_state("translucent")
        self.sphere_mesh = scene.visuals.Mesh(
            color=SPHERE_COLOR,
            shading="smooth",
            parent=self.object_node,
        )
        self.sphere_mesh.set_gl_state("translucent")
        self.vector_lines = scene.visuals.Line(
            width=VECTOR_LINE_WIDTH,
            method="gl",
            parent=self.object_node,
        )
        self.vector_tips = scene.visuals.Markers(parent=self.object_node)

        self.native = self.canvas.native
        layout.addWidget(self.native, 0, 0)

        self.vector_legend = QtWidgets.QLabel(self)
        self.vector_legend.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.vector_legend.setTextFormat(QtCore.Qt.RichText)
        self.vector_legend.setMargin(10)
        self.vector_legend.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.vector_legend.setStyleSheet(
            "QLabel {"
            "background: rgba(255, 255, 255, 215);"
            "border: 1px solid rgba(70, 80, 95, 110);"
            "border-radius: 8px;"
            "color: rgb(28, 34, 42);"
            "}"
        )
        layout.addWidget(self.vector_legend, 0, 0, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

        self.sphere_legend = QtWidgets.QLabel(self)
        self.sphere_legend.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.sphere_legend.setTextFormat(QtCore.Qt.RichText)
        self.sphere_legend.setMargin(10)
        self.sphere_legend.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.sphere_legend.setStyleSheet(
            "QLabel {"
            "background: rgba(255, 255, 255, 215);"
            "border: 1px solid rgba(70, 80, 95, 110);"
            "border-radius: 8px;"
            "color: rgb(28, 34, 42);"
            "}"
        )
        layout.addWidget(self.sphere_legend, 0, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)

        self.mask_legend = QtWidgets.QLabel(self)
        self.mask_legend.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.mask_legend.setTextFormat(QtCore.Qt.RichText)
        self.mask_legend.setMargin(10)
        self.mask_legend.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.mask_legend.setStyleSheet(
            "QLabel {"
            "background: rgba(255, 255, 255, 215);"
            "border: 1px solid rgba(70, 80, 95, 110);"
            "border-radius: 8px;"
            "color: rgb(28, 34, 42);"
            "}"
        )
        layout.addWidget(self.mask_legend, 0, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.mesh_payload = None
        self.current_palette = PALETTES[0]
        self.current_box_half_extents = None
        self.current_pixel_size_angstrom = 1.0
        self.lighting_enabled = False
        self.vectors_visible = True
        self.active_vector_set_name = "I1"
        self.sphere_target_key = None
        self.sphere_magnitude_fraction = 0.0
        self.sphere_radius_angstrom = 25.0
        self.sphere_center_xyz = None
        self.sphere_template_vertices = None
        self.sphere_outline_template_vertices = None
        self.sphere_template_faces = None
        self.sphere_template_radius = None
        self.rotation_axis = "z"
        self.rotation_angles = {axis_name: 0.0 for axis_name in ("x", "y", "z")}
        self.spin_angle = 0.0
        self.coloring_method = "radial_sphere"
        self.gradient_start = None
        self.gradient_stop = None
        self.mask_overlay_lines = []
        self._update_vector_legend()
        self._update_sphere_legend()
        self._update_mask_legend()
        self.vector_legend.hide()
        self.sphere_legend.hide()
        self.mask_legend.hide()

    def set_mesh(self, payload, palette):
        self.mesh_payload = payload
        self.current_box_half_extents = np.asarray(payload["box_half_extents_angstrom"], dtype=np.float32)
        self.current_pixel_size_angstrom = float(payload.get("pixel_size_angstrom", 1.0))
        self.current_palette = palette
        self.mesh.visible = True
        display_verts = display_space_points(payload["verts"])
        vertex_colors = palette_colors(
            radial_vertex_values(payload["verts"], method=self.coloring_method),
            palette,
            gradient_start=self.gradient_start,
            gradient_stop=self.gradient_stop,
        )
        self.mesh.set_data(
            vertices=display_verts,
            faces=payload["faces"],
            vertex_colors=vertex_colors,
        )
        self._update_mask_legend()
        self._update_vectors(payload["box_half_extents_angstrom"])
        self._update_sphere()
        self.mesh.shading = "smooth" if self.lighting_enabled else None
        self._fit_camera(display_verts)
        self.canvas.update()

    def recolor(self, palette):
        self.current_palette = palette
        if self.mesh_payload is None:
            return
        display_verts = display_space_points(self.mesh_payload["verts"])
        vertex_colors = palette_colors(
            radial_vertex_values(self.mesh_payload["verts"], method=self.coloring_method),
            palette,
            gradient_start=self.gradient_start,
            gradient_stop=self.gradient_stop,
        )
        self.mesh.set_data(
            vertices=display_verts,
            faces=self.mesh_payload["faces"],
            vertex_colors=vertex_colors,
        )
        self.canvas.update()

    def set_gradient_limits(self, gradient_start=None, gradient_stop=None):
        self.gradient_start = gradient_start
        self.gradient_stop = gradient_stop
        self.recolor(self.current_palette)

    def set_coloring_method(self, coloring_method):
        self.coloring_method = coloring_method
        self.recolor(self.current_palette)

    def set_mask_overlay(self, lines):
        self.mask_overlay_lines = [] if not lines else [str(line) for line in lines]
        self._update_mask_legend()

    def set_lighting(self, enabled):
        self.lighting_enabled = enabled
        self.mesh.shading = "smooth" if enabled else None
        self.canvas.update()

    def set_vectors_visible(self, enabled):
        self.vectors_visible = bool(enabled)
        self._apply_vector_visibility()
        self.canvas.update()

    def set_vector_set(self, vector_set_name):
        if vector_set_name not in VECTOR_SET_DEFINITIONS:
            return
        self.active_vector_set_name = vector_set_name
        if self.mesh_payload is not None:
            self._update_vectors(self.mesh_payload["box_half_extents_angstrom"])
            self._update_sphere()
        else:
            self._apply_vector_visibility()
        self.canvas.update()

    def set_sphere_target(self, target_key):
        self.sphere_target_key = None if target_key is None else str(target_key)
        self._update_sphere()
        self.canvas.update()

    def set_sphere_magnitude_fraction(self, fraction):
        self.sphere_magnitude_fraction = float(np.clip(fraction, 0.0, 1.0))
        self._update_sphere()
        self.canvas.update()

    def set_sphere_radius(self, radius_angstrom):
        self.sphere_radius_angstrom = max(float(radius_angstrom), 0.1)
        self._update_sphere()
        self.canvas.update()

    def get_sphere_max_magnitude(self):
        if self.current_box_half_extents is None:
            return 0.0
        return float(np.max(self.current_box_half_extents))

    def get_sphere_state(self):
        minimum_box_angstrom = float(self.sphere_radius_angstrom) * 3.0
        minimum_box_pixels = minimum_box_angstrom / max(float(self.current_pixel_size_angstrom), 1e-6)
        suggested_box_pixels = int(np.ceil(minimum_box_pixels / 50.0) * 50.0)
        return {
            "visible": bool(self.sphere_mesh.visible),
            "target_key": self.sphere_target_key,
            "center_xyz": None if self.sphere_center_xyz is None else np.array(self.sphere_center_xyz, copy=True),
            "radius_angstrom": float(self.sphere_radius_angstrom),
            "max_magnitude_angstrom": self.get_sphere_max_magnitude(),
            "magnitude_angstrom": float(self.sphere_magnitude_fraction) * self.get_sphere_max_magnitude(),
            "suggested_box_pixels": suggested_box_pixels,
        }

    def rotate_to(self, azimuth):
        self.spin_angle = azimuth % 360.0
        self._apply_rotation()

    def set_manual_rotation(self, axis_name, angle):
        self.rotation_angles[axis_name] = float(angle) % 360.0
        self._apply_rotation()

    def _apply_rotation(self):
        transform = MatrixTransform()
        axis_vectors = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        angles = dict(self.rotation_angles)
        angles[self.rotation_axis] = (angles[self.rotation_axis] + self.spin_angle) % 360.0
        for axis_name in ("x", "y", "z"):
            transform.rotate(angles[axis_name], axis_vectors[axis_name])
        self.object_node.transform = transform
        self.canvas.update()

    def reset_orientation(self):
        self.view.camera.elevation = self.default_elevation
        self.view.camera.azimuth = self.default_azimuth
        self.rotation_angles = {axis_name: 0.0 for axis_name in ("x", "y", "z")}
        self.spin_angle = 0.0
        self._apply_rotation()

    def set_rotation_axis(self, axis_name):
        self.rotation_axis = axis_name
        self._apply_rotation()

    def show_message(self, text):
        self.mesh.visible = False
        self.mesh_payload = None
        self.current_box_half_extents = None
        self.sphere_center_xyz = None
        self.sphere_outline_mesh.visible = False
        self.sphere_mesh.visible = False
        self.vector_lines.visible = False
        self.vector_tips.visible = False
        self._apply_vector_visibility()
        self._update_sphere_legend()
        self._update_mask_legend()
        self.canvas.update()

    def _fit_camera(self, verts):
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2.0
        span = np.max(maxs - mins)
        span = max(float(span), 1.0)
        base_scale = span * 1.6
        base_distance = span * 2.5
        self.view.camera.center = tuple(center.tolist())
        self.view.camera.scale_factor = base_scale
        self.view.camera.distance = base_distance
        self.view.camera.set_zoom_limits(
            min_scale=base_scale * 0.35,
            max_scale=base_scale * 4.0,
            min_distance=base_distance * 0.4,
            max_distance=base_distance * 6.0,
        )

    def _update_vectors(self, box_half_extents):
        vector_set = VECTOR_SET_DEFINITIONS[self.active_vector_set_name]
        box_half_extents = np.asarray(box_half_extents, dtype=np.float32) * 0.9
        safe_half_extents = np.maximum(box_half_extents, 1.0)
        vector_segments = []
        line_colors = []
        tip_points = []
        tip_colors = []
        line_connect = []
        for index, spec in enumerate(VECTOR_SPECS):
            direction = vector_set[spec["key"]]
            direction = direction / np.linalg.norm(direction)
            active_axes = np.abs(direction) > 0.0
            direction_scale = np.min(safe_half_extents[active_axes] / np.abs(direction[active_axes]))
            points = np.vstack(
                (
                    np.zeros(3, dtype=np.float32),
                    direction * direction_scale,
                )
            )
            points = display_space_points(points)
            base_index = index * 2
            vector_segments.append(points)
            line_colors.append(np.repeat(spec["color"][None, :], 2, axis=0))
            tip_points.append(points[1:2])
            tip_colors.append(spec["color"][None, :])
            line_connect.append((base_index, base_index + 1))
        all_vector_points = np.vstack(vector_segments)
        line_colors = np.vstack(line_colors)
        line_connect = np.array(line_connect, dtype=np.uint32)
        tip_points = np.vstack(tip_points)
        tip_colors = np.vstack(tip_colors)
        self.vector_lines.set_data(
            pos=all_vector_points,
            color=line_colors,
            connect=line_connect,
            width=VECTOR_LINE_WIDTH,
        )
        self.vector_tips.set_data(
            pos=tip_points,
            face_color=tip_colors,
            edge_color=tip_colors,
            size=VECTOR_MARKER_SIZE,
        )
        self._apply_vector_visibility()

    def _apply_vector_visibility(self):
        visible = self.vectors_visible and self.mesh_payload is not None
        self.vector_lines.visible = visible
        self.vector_tips.visible = visible
        self.vector_legend.setVisible(visible)
        self._update_vector_legend()
        self._update_sphere_legend()

    def _update_vector_legend(self):
        lines = [f"<b>Convention:</b> {self.active_vector_set_name}"]
        for spec in VECTOR_SPECS:
            color = spec["color"]
            rgb = tuple(int(round(float(channel) * 255.0)) for channel in color[:3])
            swatch = (
                f"<span style=\"color: rgb{rgb};\">&#9632;</span>"
            )
            lines.append(f"{swatch} {spec['label']} ({spec['color_name']})")
        self.vector_legend.setText("<br>".join(lines))

    def _update_sphere_legend(self):
        state = self.get_sphere_state()
        visible = state["visible"] and state["center_xyz"] is not None
        self.sphere_legend.setVisible(visible)
        if not visible:
            self.sphere_legend.setText("")
            return
        label = next(
            (spec["label"] for spec in VECTOR_SPECS if spec["key"] == state["target_key"]),
            str(state["target_key"]),
        )
        center = np.asarray(state["center_xyz"], dtype=np.float32)
        lines = [
            f"<b>{label}:</b> X {center[0]:.3f}, Y {center[1]:.3f}, Z {center[2]:.3f}",
            f"<b>Radius:</b> {state['radius_angstrom']:.0f} A",
            (
                f"<b>Suggested box:</b> {state['suggested_box_pixels']} px, "
                f"assuming pixel size {self.current_pixel_size_angstrom:.2f} A"
            ),
        ]
        self.sphere_legend.setText("<br>".join(lines))

    def _update_mask_legend(self):
        visible = bool(self.mask_overlay_lines) and self.mesh_payload is not None
        self.mask_legend.setVisible(visible)
        self.mask_legend.setText("" if not visible else "<br>".join(self.mask_overlay_lines))

    def _update_sphere(self):
        if self.mesh_payload is None or self.current_box_half_extents is None or self.sphere_target_key is None:
            self.sphere_center_xyz = None
            self.sphere_outline_mesh.visible = False
            self.sphere_mesh.visible = False
            self._update_sphere_legend()
            return
        vector_set = VECTOR_SET_DEFINITIONS[self.active_vector_set_name]
        direction = np.asarray(vector_set[self.sphere_target_key], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            self.sphere_center_xyz = None
            self.sphere_outline_mesh.visible = False
            self.sphere_mesh.visible = False
            self._update_sphere_legend()
            return
        direction = direction / norm
        magnitude_angstrom = self.sphere_magnitude_fraction * self.get_sphere_max_magnitude()
        center_xyz = direction * magnitude_angstrom
        self.sphere_center_xyz = center_xyz.astype(np.float32)

        if self.sphere_template_radius != self.sphere_radius_angstrom or self.sphere_template_vertices is None:
            sphere_data = create_sphere(
                rows=SPHERE_ROWS,
                cols=SPHERE_COLS,
                radius=self.sphere_radius_angstrom,
                method="latitude",
            )
            self.sphere_template_vertices = sphere_data.get_vertices().astype(np.float32)
            self.sphere_template_faces = sphere_data.get_faces().astype(np.uint32)
            outline_scale = (self.sphere_radius_angstrom + SPHERE_OUTLINE_GROWTH_ANGSTROM) / self.sphere_radius_angstrom
            self.sphere_outline_template_vertices = self.sphere_template_vertices * outline_scale
            self.sphere_template_radius = self.sphere_radius_angstrom
        display_center = display_space_points(center_xyz[np.newaxis, :])[0]
        sphere_outline_vertices = self.sphere_outline_template_vertices + display_center
        sphere_vertices = self.sphere_template_vertices + display_center
        self.sphere_outline_mesh.set_data(
            vertices=sphere_outline_vertices,
            faces=self.sphere_template_faces,
            color=SPHERE_OUTLINE_COLOR,
        )
        self.sphere_mesh.set_data(
            vertices=sphere_vertices,
            faces=self.sphere_template_faces,
            color=SPHERE_COLOR,
        )
        self.sphere_outline_mesh.visible = True
        self.sphere_mesh.visible = True
        self._update_sphere_legend()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRC Isosurface Rotation Viewer (VisPy)")
        self.resize(980, 720)

        self.volume = None
        self.pixel_size_angstrom = 1.0
        self.current_file = None
        self.mesh_process = None
        self.mesh_stdout_buffer = ""
        self.mesh_output_path = None
        self.worker_ready = False
        self.build_in_progress = False
        self.active_job = None
        self.pending_build_spec = None
        self.suppress_worker_exit_log = False
        self.mask_helper_processes = []
        self.pending_mask_overlay_info = None
        self.mesh_generation = 0
        self.applied_generation = 0
        self.pending_remesh = False
        self.rotation_angle = 0.0
        self.rotation_axis = "z"
        self.has_mesh = False

        self._build_ui()
        self._build_timers()
        self._ensure_mesh_worker()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(main_splitter)

        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)
        controls_container.setMinimumWidth(300)
        controls_container.setMaximumWidth(320)

        file_group = QtWidgets.QGroupBox("File")
        file_layout = QtWidgets.QFormLayout(file_group)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setHorizontalSpacing(6)
        file_layout.setVerticalSpacing(6)
        file_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("Choose an MRC file")
        self.file_edit.returnPressed.connect(self.load_selected_mrc)
        open_button = QtWidgets.QPushButton("Open")
        open_button.clicked.connect(self.choose_mrc)
        mask_button = QtWidgets.QPushButton("Mask Tool")
        mask_button.clicked.connect(self.open_mask_helper)
        file_row = QtWidgets.QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(6)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(open_button)
        file_row.addWidget(mask_button)
        file_row_widget = QtWidgets.QWidget()
        file_row_widget.setLayout(file_row)
        file_layout.addRow("MRC path", file_row_widget)
        controls_layout.addWidget(file_group)

        render_group = QtWidgets.QGroupBox("Surface")
        render_layout = QtWidgets.QFormLayout(render_group)
        render_layout.setContentsMargins(8, 8, 8, 8)
        render_layout.setHorizontalSpacing(6)
        render_layout.setVerticalSpacing(6)
        render_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.threshold_spin.valueChanged.connect(self.schedule_remesh)

        self.palette_combo = QtWidgets.QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.currentTextChanged.connect(self.on_palette_changed)

        self.coloring_combo = QtWidgets.QComboBox()
        for label, value in COLORING_METHODS:
            self.coloring_combo.addItem(label, value)
        self.coloring_combo.setCurrentIndex(0)
        self.coloring_combo.currentIndexChanged.connect(self.on_coloring_method_changed)

        self.step_size_spin = QtWidgets.QSpinBox()
        self.step_size_spin.setRange(1, 8)
        self.step_size_spin.setValue(2)
        self.step_size_spin.valueChanged.connect(self.schedule_remesh)

        self.display_size_combo = QtWidgets.QComboBox()
        self.display_size_combo.addItems(["96", "128", "160", "192", "256", "320", "384", "512"])
        self.display_size_combo.setCurrentText("160")
        self.display_size_combo.currentTextChanged.connect(self.schedule_remesh)

        self.gradient_start_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_start_spin.setDecimals(2)
        self.gradient_start_spin.setSingleStep(1.0)
        self.gradient_start_spin.setRange(0.0, 100000.0)
        self.gradient_start_spin.setSpecialValueText("Auto")
        self.gradient_start_spin.setSuffix(" angstrom")
        self.gradient_start_spin.setValue(0.0)
        self.gradient_start_spin.valueChanged.connect(self.update_gradient_limits)

        self.gradient_stop_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_stop_spin.setDecimals(2)
        self.gradient_stop_spin.setSingleStep(1.0)
        self.gradient_stop_spin.setRange(0.0, 100000.0)
        self.gradient_stop_spin.setSpecialValueText("Auto")
        self.gradient_stop_spin.setSuffix(" angstrom")
        self.gradient_stop_spin.setValue(0.0)
        self.gradient_stop_spin.valueChanged.connect(self.update_gradient_limits)

        render_layout.addRow("Threshold", self.threshold_spin)
        render_layout.addRow("Palette", self.palette_combo)
        render_layout.addRow("Coloring", self.coloring_combo)
        render_layout.addRow("Mesh step", self.step_size_spin)
        render_layout.addRow("Display size", self.display_size_combo)
        render_layout.addRow("Gradient start", self.gradient_start_spin)
        render_layout.addRow("Gradient stop", self.gradient_stop_spin)
        controls_layout.addWidget(render_group)

        motion_group = QtWidgets.QGroupBox("Rotation")
        motion_layout = QtWidgets.QFormLayout(motion_group)
        motion_layout.setContentsMargins(8, 8, 8, 8)
        motion_layout.setHorizontalSpacing(6)
        motion_layout.setVerticalSpacing(6)
        motion_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.rotation_button = QtWidgets.QPushButton("Start Rotation")
        self.rotation_button.clicked.connect(self.toggle_rotation)
        self.reset_orientation_button = QtWidgets.QPushButton("Reset Orientation")
        self.reset_orientation_button.clicked.connect(self.reset_orientation)
        self.rotation_axis_group = QtWidgets.QWidget()
        self.rotation_axis_layout = QtWidgets.QHBoxLayout(self.rotation_axis_group)
        self.rotation_axis_layout.setContentsMargins(0, 0, 0, 0)
        self.rotation_axis_layout.setSpacing(8)
        self.rotation_axis_buttons = {}
        for axis_name in ("x", "y", "z"):
            button = QtWidgets.QRadioButton(axis_name)
            button.toggled.connect(
                lambda checked, axis_name=axis_name: checked and self.set_rotation_axis(axis_name)
            )
            self.rotation_axis_layout.addWidget(button)
            self.rotation_axis_buttons[axis_name] = button
        self.lighting_checkbox = QtWidgets.QCheckBox("Enable lighting")
        self.lighting_checkbox.setChecked(False)
        self.lighting_checkbox.toggled.connect(self.toggle_lighting)
        self.show_vectors_checkbox = QtWidgets.QCheckBox("Show vectors")
        self.show_vectors_checkbox.setChecked(True)
        self.show_vectors_checkbox.toggled.connect(self.toggle_vectors)
        self.vector_set_combo = QtWidgets.QComboBox()
        self.vector_set_combo.addItems(VECTOR_SET_DEFINITIONS.keys())
        self.vector_set_combo.setCurrentText("I1")
        self.vector_set_combo.currentTextChanged.connect(self.set_vector_set)

        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 20.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(0.3)
        self.speed_spin.setSuffix(" deg/frame")
        self.rotation_sliders = {}
        self.rotation_value_labels = {}
        for axis_name in ("x", "y", "z"):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-12, 12)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            slider.setTickInterval(1)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            label = QtWidgets.QLabel("0 deg")
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance("-180 deg"))
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            row_layout.addWidget(slider, stretch=1)
            row_layout.addWidget(label)
            slider.valueChanged.connect(
                lambda value, axis_name=axis_name: self.on_manual_rotation_changed(axis_name, value)
            )
            self.rotation_sliders[axis_name] = slider
            self.rotation_value_labels[axis_name] = label
            motion_layout.addRow(f"{axis_name.upper()} rot", row_widget)

        motion_layout.addRow(self.rotation_button)
        motion_layout.addRow(self.reset_orientation_button)
        motion_layout.addRow("Axis", self.rotation_axis_group)
        motion_layout.addRow(self.lighting_checkbox)
        motion_layout.addRow(self.show_vectors_checkbox)
        motion_layout.addRow("Vector set", self.vector_set_combo)
        motion_layout.addRow("Speed", self.speed_spin)
        controls_layout.addWidget(motion_group)

        sphere_group = QtWidgets.QGroupBox("Sphere")
        sphere_layout = QtWidgets.QFormLayout(sphere_group)
        sphere_layout.setContentsMargins(8, 8, 8, 8)
        sphere_layout.setHorizontalSpacing(6)
        sphere_layout.setVerticalSpacing(6)
        sphere_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.sphere_target_combo = QtWidgets.QComboBox()
        self.sphere_target_combo.addItem("None", None)
        for spec in VECTOR_SPECS:
            self.sphere_target_combo.addItem(spec["label"], spec["key"])
        self.sphere_target_combo.currentIndexChanged.connect(self.on_sphere_target_changed)

        self.sphere_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sphere_distance_slider.setRange(0, 1000)
        self.sphere_distance_slider.setValue(0)
        self.sphere_distance_slider.valueChanged.connect(self.on_sphere_distance_changed)
        self.sphere_distance_slider.sliderReleased.connect(self.on_sphere_distance_released)
        self.sphere_distance_label = QtWidgets.QLabel("0.0 / 0.0 A")
        self.sphere_distance_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        sphere_distance_widget = QtWidgets.QWidget()
        sphere_distance_layout = QtWidgets.QVBoxLayout(sphere_distance_widget)
        sphere_distance_layout.setContentsMargins(0, 0, 0, 0)
        sphere_distance_layout.setSpacing(2)
        sphere_distance_layout.addWidget(self.sphere_distance_slider)
        sphere_distance_layout.addWidget(self.sphere_distance_label)

        self.sphere_radius_spin = QtWidgets.QDoubleSpinBox()
        self.sphere_radius_spin.setRange(1.0, 100000.0)
        self.sphere_radius_spin.setDecimals(0)
        self.sphere_radius_spin.setSingleStep(1.0)
        self.sphere_radius_spin.setSuffix(" A")
        self.sphere_radius_spin.setValue(25.0)
        self.sphere_radius_spin.valueChanged.connect(self.on_sphere_radius_changed)

        sphere_layout.addRow("Target", self.sphere_target_combo)
        sphere_layout.addRow("Distance", sphere_distance_widget)
        sphere_layout.addRow("Radius", self.sphere_radius_spin)
        controls_layout.addWidget(sphere_group)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumBlockCount(200)
        self.status_box.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.status_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.status_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.status_box.setMaximumHeight(130)
        controls_layout.addWidget(self.status_box, stretch=1)
        controls_layout.addStretch(0)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        viewer_panel = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = VispyIsosurfaceWidget()
        self.viewer.set_rotation_axis(self.rotation_axis)
        viewer_layout.addWidget(self.viewer)

        sections_panel = QtWidgets.QGroupBox("Central Sections")
        sections_layout = QtWidgets.QHBoxLayout(sections_panel)
        sections_layout.setContentsMargins(8, 8, 8, 8)
        sections_layout.setSpacing(10)
        self.slice_views = {
            "xy": SliceView("XY"),
            "yz": SliceView("YZ"),
            "xz": SliceView("XZ"),
        }
        for key in ("xy", "yz", "xz"):
            sections_layout.addWidget(self.slice_views[key], stretch=1)

        self.rotation_axis_buttons[self.rotation_axis].setChecked(True)

        right_splitter.addWidget(viewer_panel)
        right_splitter.addWidget(sections_panel)
        right_splitter.setSizes([620, 260])

        main_splitter.addWidget(controls_container)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([310, 970])

    def _build_timers(self):
        self.draft_remesh_timer = QtCore.QTimer(self)
        self.draft_remesh_timer.setSingleShot(True)
        self.draft_remesh_timer.setInterval(180)
        self.draft_remesh_timer.timeout.connect(lambda: self.start_remesh(final=False))

        self.final_remesh_timer = QtCore.QTimer(self)
        self.final_remesh_timer.setSingleShot(True)
        self.final_remesh_timer.setInterval(900)
        self.final_remesh_timer.timeout.connect(lambda: self.start_remesh(final=True))

        self.rotation_timer = QtCore.QTimer(self)
        self.rotation_timer.setInterval(16)
        self.rotation_timer.timeout.connect(self.advance_rotation)

    def _ensure_mesh_worker(self):
        if self.mesh_process is not None:
            return
        self.mesh_process = QtCore.QProcess(self)
        self.mesh_stdout_buffer = ""
        self.worker_ready = False
        self.mesh_process.setProgram(sys.executable)
        self.mesh_process.setArguments([str(REPO_DIR / "mrc_isosurface_vispy_mesh_worker.py")])
        self.mesh_process.setWorkingDirectory(str(REPO_DIR))
        self.mesh_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.mesh_process.readyReadStandardOutput.connect(self._drain_mesh_output)
        self.mesh_process.finished.connect(self._mesh_worker_exited)
        self.mesh_process.start()

    def _restart_mesh_worker(self):
        proc = self.mesh_process
        if proc is not None:
            try:
                self.suppress_worker_exit_log = True
                proc.kill()
                proc.waitForFinished(1000)
            except Exception:
                pass
            if self.mesh_process is proc:
                proc.deleteLater()
        self.mesh_process = None
        self.worker_ready = False
        self.build_in_progress = False
        self.active_job = None
        self.pending_build_spec = None
        self._ensure_mesh_worker()

    def choose_mrc(self):
        was_rotating = self.rotation_timer.isActive()
        self.pause_for_file_dialog()
        start_path = str(REPO_DIR)
        if self.current_file is not None:
            start_path = str(self.current_file.parent)
        elif self.file_edit.text().strip():
            typed_path = Path(self.file_edit.text().strip())
            start_path = str(typed_path.parent if typed_path.suffix else typed_path)

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose MRC file",
            start_path,
            "MRC files (*.mrc *.map *.mrcs);;All files (*)",
        )
        if not path:
            self.resume_rotation_if_needed(was_rotating)
            return
        self.prepare_for_new_mrc()
        self.file_edit.setText(path)
        self.load_selected_mrc()

    def load_selected_mrc(self):
        path = self.file_edit.text().strip()
        if not path:
            return
        mrc_path = Path(path)
        if self.current_file is not None and mrc_path != self.current_file:
            self.prepare_for_new_mrc()
        if not mrc_path.is_file():
            self.log(f"File not found: {mrc_path}")
            QtWidgets.QMessageBox.critical(self, "Open Failed", f"File not found:\n{mrc_path}")
            return

        try:
            self.volume, self.pixel_size_angstrom = load_volume(mrc_path)
        except Exception as exc:
            self.log(f"Failed to load volume: {exc}")
            QtWidgets.QMessageBox.critical(self, "Open Failed", str(exc))
            return

        self.current_file = mrc_path
        overlay_info = None
        if self.pending_mask_overlay_info is not None:
            pending_path = Path(self.pending_mask_overlay_info.get("path", "")).expanduser()
            if pending_path == mrc_path:
                overlay_info = self.pending_mask_overlay_info.get("mask_summary")
        self.viewer.set_mask_overlay(overlay_info)
        self.pending_mask_overlay_info = None
        threshold = default_threshold(self.volume)
        vol_min = float(np.min(self.volume))
        vol_max = float(np.max(self.volume))
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setRange(vol_min, vol_max)
        self.threshold_spin.setValue(min(max(threshold, vol_min), vol_max))
        self.threshold_spin.blockSignals(False)

        self.log(
            f"Loaded {mrc_path.name}: shape={self.volume.shape}, "
            f"min={vol_min:.4f}, max={vol_max:.4f}, "
            f"pixel_size={self.pixel_size_angstrom:.4f} angstrom"
        )
        self.schedule_remesh()
        self.update_section_views()

    def open_mask_helper(self):
        if self.current_file is None:
            QtWidgets.QMessageBox.information(self, "Mask Tool", "Load an MRC file first.")
            return
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments([str(REPO_DIR / "mrc_mask_helper.py"), str(self.current_file)])
        proc.setWorkingDirectory(str(REPO_DIR))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc._stdout_buffer = ""
        proc.readyReadStandardOutput.connect(lambda proc=proc: self._drain_mask_helper_output(proc))
        proc.finished.connect(lambda _code, _status, proc=proc: self._mask_helper_exited(proc))
        proc.start()
        if not proc.waitForStarted(1000):
            QtWidgets.QMessageBox.critical(self, "Mask Tool", "Failed to launch mask helper.")
            proc.deleteLater()
            return
        self.mask_helper_processes.append(proc)

    def _drain_mask_helper_output(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        proc._stdout_buffer += data
        while "\n" in proc._stdout_buffer:
            line, proc._stdout_buffer = proc._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_mask_helper_message(line)

    def _handle_mask_helper_message(self, line):
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self.log(line)
            return
        if message.get("type") != "load_masked_output":
            return
        masked_path = Path(message.get("path", "")).expanduser()
        if not masked_path.is_file():
            self.log(f"Masked output not found for auto-load: {masked_path}")
            return
        self.pending_mask_overlay_info = {
            "path": str(masked_path),
            "mask_summary": message.get("mask_summary", []),
        }
        self.file_edit.setText(str(masked_path))
        self.load_selected_mrc()
        self.log(f"Auto-loaded masked output: {masked_path.name}")

    def _mask_helper_exited(self, proc):
        if proc in self.mask_helper_processes:
            self.mask_helper_processes.remove(proc)
        proc.deleteLater()

    def prepare_for_new_mrc(self):
        if self.current_file is None and self.volume is None and not self.has_mesh:
            return
        self.rotation_timer.stop()
        self.rotation_button.setText("Start Rotation")
        self.has_mesh = False
        self.rotation_angle = 0.0
        self._reset_rotation_sliders()
        self.viewer.set_mask_overlay(None)
        self.pending_remesh = False
        self.draft_remesh_timer.stop()
        self.final_remesh_timer.stop()
        self._restart_mesh_worker()
        self.mesh_generation += 1
        self.applied_generation = self.mesh_generation
        self.viewer.reset_orientation()
        self.viewer.show_message("Loading new MRC...")
        self.clear_section_views("No data")
        self.volume = None
        self.pixel_size_angstrom = 1.0
        self.current_file = None
        self.log("Paused rotation and cleared current MRC before opening a new one.")

    def pause_for_file_dialog(self):
        if self.has_mesh:
            self.rotation_timer.stop()
            self.rotation_button.setText("Start Rotation")
            self.log("Paused rotation while opening file chooser.")

    def resume_rotation_if_needed(self, was_rotating):
        if was_rotating and self.has_mesh:
            self.rotation_timer.start()
            self.rotation_button.setText("Stop Rotation")
            self.log("Resumed rotation after file chooser closed.")

    def schedule_remesh(self, *_args):
        if self.volume is None:
            return
        self.draft_remesh_timer.start()
        self.final_remesh_timer.start()

    def _build_spec(self, final):
        threshold = float(self.threshold_spin.value())
        selected_step = int(self.step_size_spin.value())
        selected_display = int(self.display_size_combo.currentText())
        if final:
            return {
                "quality": "final",
                "threshold": threshold,
                "step_size": selected_step,
                "display_size": selected_display,
            }
        return {
            "quality": "draft",
            "threshold": threshold,
            "step_size": max(selected_step, 3),
            "display_size": min(selected_display, 128),
        }

    def start_remesh(self, final):
        if self.volume is None:
            return
        self._ensure_mesh_worker()
        spec = self._build_spec(final=final)
        if self.build_in_progress or not self.worker_ready:
            existing = self.pending_build_spec
            if existing is None or (existing["quality"] == "draft" and spec["quality"] == "final"):
                self.pending_build_spec = spec
            return
        self.mesh_generation += 1
        generation = self.mesh_generation
        self.log(
            f"Building {spec['quality']} mesh: threshold={spec['threshold']:.4f}, "
            f"step_size={spec['step_size']}, display_size={spec['display_size']}"
        )
        output_dir = REPO_DIR / ".vispy_mesh_cache"
        output_dir.mkdir(exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            suffix=".npz",
            prefix="mesh-",
            dir=output_dir,
            delete=False,
        )
        handle.close()
        self.mesh_output_path = handle.name
        self.active_job = {
            "job_id": generation,
            "quality": spec["quality"],
            "output_npz": self.mesh_output_path,
        }
        self.build_in_progress = True
        command = {
            "cmd": "build",
            "job_id": generation,
            "input_mrc": str(self.current_file),
            "threshold": spec["threshold"],
            "step_size": spec["step_size"],
            "display_size": spec["display_size"],
            "output_npz": self.mesh_output_path,
        }
        self.mesh_process.write((json.dumps(command) + "\n").encode("utf-8"))

    def on_mesh_ready(self, generation, payload):
        if generation < self.applied_generation:
            return
        self.applied_generation = generation
        self.has_mesh = True
        self.viewer.set_mesh(payload, self.palette_combo.currentText())
        self.update_section_views()
        self.update_sphere_distance_label()
        self.log(
            f"Mesh ready: verts={len(payload['verts'])}, faces={len(payload['faces'])}, "
            f"threshold={payload['threshold']:.4f}"
        )
        info = payload["downsample_info"]
        self.log(
            "Downsample applied: "
            f"factor={int(info['factor'])}, "
            f"shape={tuple(int(x) for x in info['shape'])}"
        )
        self.log_sphere_state()

    def on_mesh_failed(self, generation, message):
        if generation >= self.applied_generation:
            self.log(f"Mesh build failed: {message}")
            QtWidgets.QMessageBox.critical(self, "Mesh Build Failed", message)

    def _drain_mesh_output(self):
        if self.mesh_process is None:
            return
        data = bytes(self.mesh_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self.mesh_stdout_buffer += data
        while "\n" in self.mesh_stdout_buffer:
            line, self.mesh_stdout_buffer = self.mesh_stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_worker_message(line)

    def _handle_worker_message(self, line):
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self.log(line)
            return

        msg_type = message.get("type")
        if msg_type == "ready":
            self.worker_ready = True
            if self.pending_build_spec is not None and self.volume is not None:
                spec = self.pending_build_spec
                self.pending_build_spec = None
                QtCore.QTimer.singleShot(0, lambda spec=spec: self.start_remesh(final=(spec["quality"] == "final")))
            return
        if msg_type == "log":
            self.log(message.get("message", ""))
            return
        if msg_type == "result":
            self._handle_mesh_result(message)
            return
        if msg_type == "error":
            self._handle_mesh_error(message)
            return
        if msg_type == "cleared":
            return
        if msg_type == "shutdown":
            return

    def _handle_mesh_result(self, message):
        job_id = int(message["job_id"])
        output_npz = Path(message["output_npz"])
        if output_npz.is_file():
            data = np.load(output_npz)
            downsample_factor = int(data["downsample_factor"][0])
            downsample_shape = tuple(int(x) for x in data["downsample_shape"])
            effective_angstrom_per_voxel = float(self.pixel_size_angstrom) * float(downsample_factor)
            box_half_extents_angstrom = (
                (array_shape_to_xyz(np.array(downsample_shape, dtype=np.float32)) - 1.0) / 2.0
            ) * effective_angstrom_per_voxel
            payload = {
                "verts": data["verts"],
                "faces": data["faces"],
                "vertex_values": data["vertex_values"],
                "threshold": float(data["threshold"][0]),
                "pixel_size_angstrom": float(self.pixel_size_angstrom),
                "box_half_extents_angstrom": box_half_extents_angstrom,
                "downsample_info": {
                    "factor": downsample_factor,
                    "shape": downsample_shape,
                },
            }
            self.on_mesh_ready(job_id, payload)
            try:
                output_npz.unlink()
            except OSError:
                pass
        self.build_in_progress = False
        self.active_job = None
        self.mesh_output_path = None
        self._maybe_start_pending_build()

    def _handle_mesh_error(self, message):
        job_id = int(message.get("job_id") or -1)
        self.build_in_progress = False
        self.active_job = None
        if self.mesh_output_path:
            try:
                Path(self.mesh_output_path).unlink()
            except OSError:
                pass
            self.mesh_output_path = None
        self.on_mesh_failed(job_id, message.get("message", "Mesh build failed."))
        self._maybe_start_pending_build()

    def _maybe_start_pending_build(self):
        if self.pending_build_spec is None or self.volume is None or not self.worker_ready or self.build_in_progress:
            return
        spec = self.pending_build_spec
        self.pending_build_spec = None
        self.start_remesh(final=(spec["quality"] == "final"))

    def _mesh_worker_exited(self, exit_code, exit_status):
        self.worker_ready = False
        self.build_in_progress = False
        self.active_job = None
        if self.mesh_process is not None:
            self.mesh_process.deleteLater()
            self.mesh_process = None
        if self.suppress_worker_exit_log:
            self.suppress_worker_exit_log = False
            return
        if exit_status != QtCore.QProcess.NormalExit:
            self.log("Mesh worker process crashed; restarting.")
            self._ensure_mesh_worker()

    def on_palette_changed(self, palette):
        self.viewer.recolor(palette)
        self.update_section_views()

    def on_coloring_method_changed(self, _index):
        coloring_method = self.coloring_combo.currentData()
        self.viewer.set_coloring_method(coloring_method)
        self.update_section_views()
        label = self.coloring_combo.currentText()
        self.log(f"Coloring method: {label}")

    def update_gradient_limits(self, *_args):
        gradient_start = None if self.gradient_start_spin.value() == 0.0 else float(self.gradient_start_spin.value())
        gradient_stop = None if self.gradient_stop_spin.value() == 0.0 else float(self.gradient_stop_spin.value())
        self.viewer.set_gradient_limits(gradient_start=gradient_start, gradient_stop=gradient_stop)
        self.update_section_views()

    def toggle_rotation(self):
        if not self.has_mesh:
            self.rotation_timer.stop()
            self.rotation_button.setText("Start Rotation")
            return
        if self.rotation_timer.isActive():
            self.rotation_timer.stop()
            self.rotation_button.setText("Start Rotation")
        else:
            self.rotation_timer.start()
            self.rotation_button.setText("Stop Rotation")

    def toggle_lighting(self, enabled):
        self.viewer.set_lighting(enabled)
        self.log(f"Lighting {'enabled' if enabled else 'disabled'}")

    def toggle_vectors(self, enabled):
        self.viewer.set_vectors_visible(enabled)
        self.update_section_views()
        self.log(f"Vectors {'shown' if enabled else 'hidden'}")

    def set_vector_set(self, vector_set_name):
        if not hasattr(self, "viewer"):
            return
        self.viewer.set_vector_set(vector_set_name)
        self.update_section_views()
        self.log(f"Vector set: {vector_set_name}")
        self.log_sphere_state()

    def on_sphere_target_changed(self, index):
        target_key = self.sphere_target_combo.itemData(index)
        self.viewer.set_sphere_target(target_key)
        self.update_sphere_distance_label()
        self.update_section_views()
        self.log_sphere_state()

    def on_sphere_distance_changed(self, slider_value):
        fraction = float(slider_value) / float(self.sphere_distance_slider.maximum())
        self.viewer.set_sphere_magnitude_fraction(fraction)
        self.update_sphere_distance_label()
        self.update_section_views()

    def on_sphere_distance_released(self):
        self.log_sphere_state()

    def on_sphere_radius_changed(self, radius_angstrom):
        self.viewer.set_sphere_radius(radius_angstrom)
        self.update_section_views()
        self.log_sphere_state()

    def update_sphere_distance_label(self):
        state = self.viewer.get_sphere_state()
        magnitude = float(state["magnitude_angstrom"])
        maximum = float(state["max_magnitude_angstrom"])
        self.sphere_distance_label.setText(f"{magnitude:.1f} / {maximum:.1f} A")

    def log_sphere_state(self):
        state = self.viewer.get_sphere_state()
        if not state["visible"] or state["center_xyz"] is None:
            return
        center = np.asarray(state["center_xyz"], dtype=np.float32)
        label = next(
            (spec["label"] for spec in VECTOR_SPECS if spec["key"] == state["target_key"]),
            str(state["target_key"]),
        )
        self.log(
            f"Sphere {label}: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) A, "
            f"radius={state['radius_angstrom']:.3f} A"
        )

    def reset_orientation(self):
        self.rotation_angle = 0.0
        self._reset_rotation_sliders()
        self.viewer.reset_orientation()
        self.log("Orientation reset")

    def set_rotation_axis(self, axis_name):
        self.rotation_axis = axis_name
        if not hasattr(self, "viewer"):
            return
        self.viewer.set_rotation_axis(axis_name)
        self.viewer.rotate_to(self.rotation_angle)
        self.log(f"Rotation axis set to {axis_name}")

    def on_manual_rotation_changed(self, axis_name, slider_value):
        angle = int(slider_value) * 15
        self.rotation_value_labels[axis_name].setText(f"{angle} deg")
        if not hasattr(self, "viewer"):
            return
        self.viewer.set_manual_rotation(axis_name, angle)

    def _reset_rotation_sliders(self):
        for slider in self.rotation_sliders.values():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        for label in self.rotation_value_labels.values():
            label.setText("0 deg")

    def advance_rotation(self):
        if not self.rotation_timer.isActive() or not self.has_mesh:
            return
        self.rotation_angle = (self.rotation_angle + float(self.speed_spin.value())) % 360.0
        self.viewer.rotate_to(self.rotation_angle)

    def log(self, text):
        self.status_box.appendPlainText(text)

    def clear_section_views(self, text="No data"):
        for slice_view in self.slice_views.values():
            slice_view.clear(text)

    def update_section_views(self):
        if self.volume is None:
            self.clear_section_views("No data")
            return

        palette = self.palette_combo.currentText()
        coloring_method = self.coloring_combo.currentData()
        gradient_start = None if self.gradient_start_spin.value() == 0.0 else float(self.gradient_start_spin.value())
        gradient_stop = None if self.gradient_stop_spin.value() == 0.0 else float(self.gradient_stop_spin.value())
        threshold = float(self.threshold_spin.value())
        volume_max = float(np.max(self.volume))
        center = (np.array(self.volume.shape, dtype=np.float32) - 1.0) / 2.0
        sphere_state = self.viewer.get_sphere_state()

        section_specs = {
            "xy": {
                "slice_data": self.volume[self.volume.shape[0] // 2, :, :],
                "axes": (1, 2),
                "plane_axis_xyz": 2,
                "plane_value_angstrom": 0.0,
                "display_center_xyz_axes": (1, 0),
                "xyz_axes": ("x", "y"),
            },
            "yz": {
                "slice_data": self.volume[:, :, self.volume.shape[2] // 2],
                "axes": (0, 1),
                "plane_axis_xyz": 0,
                "plane_value_angstrom": 0.0,
                "display_center_xyz_axes": (2, 1),
                "xyz_axes": ("y", "z"),
            },
            "xz": {
                "slice_data": self.volume[:, self.volume.shape[1] // 2, :],
                "axes": (0, 2),
                "plane_axis_xyz": 1,
                "plane_value_angstrom": 0.0,
                "display_center_xyz_axes": (2, 0),
                "xyz_axes": ("x", "z"),
            },
        }

        for key, spec in section_specs.items():
            try:
                pixmap = self._build_section_pixmap(
                    slice_data=spec["slice_data"],
                    axes=spec["axes"],
                    center=center,
                    pixel_size_angstrom=self.pixel_size_angstrom,
                    palette=palette,
                    coloring_method=coloring_method,
                    gradient_start=gradient_start,
                    gradient_stop=gradient_stop,
                    threshold=threshold,
                    volume_max=volume_max,
                    sphere_state=sphere_state,
                    plane_axis_xyz=spec["plane_axis_xyz"],
                    plane_value_angstrom=spec["plane_value_angstrom"],
                    display_center_xyz_axes=spec["display_center_xyz_axes"],
                    xyz_axes=spec["xyz_axes"],
                )
            except Exception as exc:
                self.log(f"Section overlay fallback ({key}): {exc}")
                pixmap = self._build_section_pixmap(
                    slice_data=spec["slice_data"],
                    axes=spec["axes"],
                    center=center,
                    pixel_size_angstrom=self.pixel_size_angstrom,
                    palette=palette,
                    coloring_method=coloring_method,
                    gradient_start=gradient_start,
                    gradient_stop=gradient_stop,
                    threshold=threshold,
                    volume_max=volume_max,
                    sphere_state={"visible": False, "center_xyz": None, "radius_angstrom": 0.0},
                    plane_axis_xyz=spec["plane_axis_xyz"],
                    plane_value_angstrom=spec["plane_value_angstrom"],
                    display_center_xyz_axes=spec["display_center_xyz_axes"],
                    xyz_axes=spec["xyz_axes"],
                    draw_overlays=False,
                )
            self.slice_views[key].set_pixmap(pixmap)

    def _build_section_pixmap(
        self,
        slice_data,
        axes,
        center,
        pixel_size_angstrom,
        palette,
        coloring_method,
        gradient_start,
        gradient_stop,
        threshold,
        volume_max,
        sphere_state,
        plane_axis_xyz,
        plane_value_angstrom,
        display_center_xyz_axes,
        xyz_axes,
        draw_overlays=True,
    ):
        rows = np.arange(slice_data.shape[0], dtype=np.float32)
        cols = np.arange(slice_data.shape[1], dtype=np.float32)
        row_coords = (rows - center[axes[0]]) * float(pixel_size_angstrom)
        col_coords = (cols - center[axes[1]]) * float(pixel_size_angstrom)
        grid_row, grid_col = np.meshgrid(row_coords, col_coords, indexing="ij")
        xyz_grids = {
            "x": np.zeros_like(grid_row),
            "y": np.zeros_like(grid_row),
            "z": np.zeros_like(grid_row),
        }
        xyz_grids[xyz_axes[0]] = grid_col
        xyz_grids[xyz_axes[1]] = grid_row
        radial = radial_values_from_components(
            xyz_grids["x"],
            xyz_grids["y"],
            xyz_grids["z"],
            coloring_method,
        )

        colors = palette_colors(
            radial,
            palette,
            gradient_start=gradient_start,
            gradient_stop=gradient_stop,
        )[:, :, :3]

        denom = max(volume_max - threshold, 1e-6)
        opacity = np.clip((slice_data.astype(np.float32) - threshold) / denom, 0.0, 1.0)
        opacity = np.sqrt(opacity)

        background = np.array([0.96, 0.97, 0.98], dtype=np.float32)
        rgb = (background * (1.0 - opacity[..., None])) + (colors * opacity[..., None])
        rgb = np.flipud(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))
        pixmap = make_qpixmap(rgb)
        if not draw_overlays:
            return pixmap
        pixmap = self._draw_vector_section_overlay(
            pixmap=pixmap,
            plane_axis_xyz=plane_axis_xyz,
            display_center_xyz_axes=display_center_xyz_axes,
            center=center,
            pixel_size_angstrom=pixel_size_angstrom,
        )
        return self._draw_sphere_section_overlay(
            pixmap=pixmap,
            sphere_state=sphere_state,
            plane_axis_xyz=plane_axis_xyz,
            plane_value_angstrom=plane_value_angstrom,
            display_center_xyz_axes=display_center_xyz_axes,
            center=center,
            pixel_size_angstrom=pixel_size_angstrom,
        )

    def _xyz_to_section_pixel(self, xyz_point, display_center_xyz_axes, center, pixel_size_angstrom, pixmap_height):
        xyz_point = np.asarray(xyz_point, dtype=np.float32)
        row_axis_xyz, col_axis_xyz = display_center_xyz_axes
        xyz_to_array_axis = {0: 2, 1: 1, 2: 0}
        row_array_axis = xyz_to_array_axis[row_axis_xyz]
        col_array_axis = xyz_to_array_axis[col_axis_xyz]
        row_index = (float(xyz_point[row_axis_xyz]) / float(pixel_size_angstrom)) + float(center[row_array_axis])
        col_index = (float(xyz_point[col_axis_xyz]) / float(pixel_size_angstrom)) + float(center[col_array_axis])
        display_row_index = (pixmap_height - 1) - row_index
        return QtCore.QPointF(col_index, display_row_index)

    def _draw_vector_section_overlay(
        self,
        pixmap,
        plane_axis_xyz,
        display_center_xyz_axes,
        center,
        pixel_size_angstrom,
    ):
        if not self.viewer.vectors_visible or self.viewer.current_box_half_extents is None:
            return pixmap

        box_half_extents = np.asarray(self.viewer.current_box_half_extents, dtype=np.float32) * 0.9
        section_vector_length = max(float(np.max(box_half_extents)), 1.0)
        vector_set = VECTOR_SET_DEFINITIONS[self.viewer.active_vector_set_name]
        origin_point = self._xyz_to_section_pixel(
            np.zeros(3, dtype=np.float32),
            display_center_xyz_axes=display_center_xyz_axes,
            center=center,
            pixel_size_angstrom=pixel_size_angstrom,
            pixmap_height=pixmap.height(),
        )
        overlay_pixmap = QtGui.QPixmap(pixmap)
        painter = QtGui.QPainter(overlay_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        for spec in VECTOR_SPECS:
            direction = np.asarray(vector_set[spec["key"]], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                continue
            direction = direction / norm
            row_axis_xyz, col_axis_xyz = display_center_xyz_axes
            projected_direction = np.array(
                (direction[row_axis_xyz], direction[col_axis_xyz]),
                dtype=np.float32,
            )
            projected_norm = float(np.linalg.norm(projected_direction))
            rgb = tuple(int(round(float(channel) * 255.0)) for channel in spec["color"][:3])
            qcolor = QtGui.QColor(*rgb, 235)
            pen = QtGui.QPen(qcolor)
            pen.setWidthF(SECTION_VECTOR_LINE_WIDTH)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(qcolor))
            # Strict mode: only vectors lying in the section plane appear as lines.
            # Otherwise the vector intersects the central slice only at the origin.
            if abs(float(direction[plane_axis_xyz])) > 1e-6 or projected_norm <= 0.0:
                painter.drawEllipse(
                    QtCore.QRectF(
                        origin_point.x() - SECTION_VECTOR_TIP_RADIUS,
                        origin_point.y() - SECTION_VECTOR_TIP_RADIUS,
                        SECTION_VECTOR_TIP_RADIUS * 2.0,
                        SECTION_VECTOR_TIP_RADIUS * 2.0,
                    )
                )
                continue

            projected_direction = projected_direction / projected_norm
            endpoint_xyz = np.zeros(3, dtype=np.float32)
            endpoint_xyz[row_axis_xyz] = projected_direction[0] * section_vector_length
            endpoint_xyz[col_axis_xyz] = projected_direction[1] * section_vector_length
            endpoint_point = self._xyz_to_section_pixel(
                endpoint_xyz,
                display_center_xyz_axes=display_center_xyz_axes,
                center=center,
                pixel_size_angstrom=pixel_size_angstrom,
                pixmap_height=pixmap.height(),
            )
            painter.drawLine(QtCore.QLineF(origin_point, endpoint_point))
            painter.drawEllipse(
                QtCore.QRectF(
                    endpoint_point.x() - SECTION_VECTOR_TIP_RADIUS,
                    endpoint_point.y() - SECTION_VECTOR_TIP_RADIUS,
                    SECTION_VECTOR_TIP_RADIUS * 2.0,
                    SECTION_VECTOR_TIP_RADIUS * 2.0,
                )
            )

        painter.end()
        return overlay_pixmap

    def _draw_sphere_section_overlay(
        self,
        pixmap,
        sphere_state,
        plane_axis_xyz,
        plane_value_angstrom,
        display_center_xyz_axes,
        center,
        pixel_size_angstrom,
    ):
        if not sphere_state["visible"] or sphere_state["center_xyz"] is None:
            return pixmap

        sphere_center = np.asarray(sphere_state["center_xyz"], dtype=np.float32)
        sphere_radius = float(sphere_state["radius_angstrom"])
        plane_offset = float(sphere_center[plane_axis_xyz] - plane_value_angstrom)
        if abs(plane_offset) > sphere_radius:
            return pixmap

        section_radius_angstrom = float(np.sqrt(max((sphere_radius ** 2) - (plane_offset ** 2), 0.0)))
        section_radius_pixels = section_radius_angstrom / max(float(pixel_size_angstrom), 1e-6)
        if section_radius_pixels <= 0.0:
            return pixmap

        center_point = self._xyz_to_section_pixel(
            sphere_center,
            display_center_xyz_axes=display_center_xyz_axes,
            center=center,
            pixel_size_angstrom=pixel_size_angstrom,
            pixmap_height=pixmap.height(),
        )

        overlay_pixmap = QtGui.QPixmap(pixmap)
        painter = QtGui.QPainter(overlay_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(SECTION_SPHERE_OUTLINE_COLOR)
        pen.setWidthF(SECTION_SPHERE_LINE_WIDTH)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(SECTION_SPHERE_FILL_COLOR))
        painter.drawEllipse(
            QtCore.QRectF(
                center_point.x() - section_radius_pixels,
                center_point.y() - section_radius_pixels,
                section_radius_pixels * 2.0,
                section_radius_pixels * 2.0,
            )
        )
        painter.end()
        return overlay_pixmap


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
