import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
from PySide6 import QtCore, QtWidgets


def load_mrc(input_path):
    with mrcfile.open(input_path) as mrc:
        volume = np.array(mrc.data, copy=True)
        voxel_size = getattr(mrc, "voxel_size", None)
        angstrom_per_pixel = float(voxel_size.x) if voxel_size and voxel_size.x else 1.0
    return volume, angstrom_per_pixel


def write_mrc(output_path, data, angstrom_per_pixel):
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = angstrom_per_pixel
        mrc.update_header_from_data()
        mrc.update_header_stats()


def coordinate_grids(shape, angstrom_per_pixel):
    center = (np.array(shape, dtype=np.float32) - 1.0) / 2.0
    z = (np.arange(shape[0], dtype=np.float32) - center[0]) * float(angstrom_per_pixel)
    y = (np.arange(shape[1], dtype=np.float32) - center[1]) * float(angstrom_per_pixel)
    x = (np.arange(shape[2], dtype=np.float32) - center[2]) * float(angstrom_per_pixel)
    return np.meshgrid(z, y, x, indexing="ij")


def build_mask(volume_shape, angstrom_per_pixel, shape_name, dimensions, orientation):
    zz, yy, xx = coordinate_grids(volume_shape, angstrom_per_pixel)

    if shape_name == "sphere":
        radius = float(dimensions["sphere_radius"])
        z_offset = float(dimensions["sphere_z_offset"])
        return ((xx ** 2) + (yy ** 2) + ((zz - z_offset) ** 2) <= radius ** 2).astype(np.float32)

    if shape_name == "square-cube":
        half_extent = float(dimensions["cube_x"]) / 2.0
        return (
            (np.abs(xx) <= half_extent)
            & (np.abs(yy) <= half_extent)
            & (np.abs(zz) <= half_extent)
        ).astype(np.float32)

    if shape_name == "cylinder":
        radius = float(dimensions["cylinder_radius"])
        height = float(dimensions["cylinder_height"])
        axis_map = {
            "x": (yy ** 2) + (zz ** 2),
            "y": (xx ** 2) + (zz ** 2),
            "z": (xx ** 2) + (yy ** 2),
        }
        height_map = {
            "x": np.abs(xx),
            "y": np.abs(yy),
            "z": np.abs(zz),
        }
        radial = axis_map[orientation]
        axial = height_map[orientation]
        return ((radial <= radius ** 2) & (axial <= (height / 2.0))).astype(np.float32)

    if shape_name == "rectangle-cube":
        half_x = float(dimensions["cube_x"]) / 2.0
        half_y = float(dimensions["cube_y"]) / 2.0
        half_z = float(dimensions["cube_z"]) / 2.0
        return (
            (np.abs(xx) <= half_x)
            & (np.abs(yy) <= half_y)
            & (np.abs(zz) <= half_z)
        ).astype(np.float32)

    raise ValueError(f"Unsupported shape: {shape_name}")


def mask_summary(shape_name, dimensions, orientation):
    display_names = {
        "sphere": "Sphere",
        "square-cube": "Square-Cube",
        "cylinder": "Cylinder",
        "rectangle-cube": "Rectangle-Cube",
    }
    parts = [f"Mask: {display_names.get(shape_name, shape_name)}"]
    if shape_name == "sphere":
        parts.append(f"Radius {float(dimensions['sphere_radius']):.2f} A")
        parts.append(f"Z offset {float(dimensions['sphere_z_offset']):.2f} A")
    elif shape_name == "square-cube":
        parts.append(f"Edge {float(dimensions['cube_x']):.2f} A")
    elif shape_name == "cylinder":
        parts.append(f"Radius {float(dimensions['cylinder_radius']):.2f} A")
        parts.append(f"Height {float(dimensions['cylinder_height']):.2f} A")
        parts.append(f"Axis {orientation.upper()}")
    elif shape_name == "rectangle-cube":
        parts.append(f"Lx {float(dimensions['cube_x']):.2f} A")
        parts.append(f"Ly {float(dimensions['cube_y']):.2f} A")
        parts.append(f"Lz {float(dimensions['cube_z']):.2f} A")
    return parts


class MrcMaskHelper(QtWidgets.QWidget):
    def __init__(self, input_path=None, defaults=None):
        super().__init__()
        self.setWindowTitle("MRC Mask Helper")
        self.resize(520, 0)
        self.volume = None
        self.angstrom_per_pixel = 1.0
        self.defaults = defaults or {}
        self._build_ui()
        if input_path is not None:
            self.input_edit.setText(str(input_path))
            self.load_input_mrc()
        self._update_shape_controls()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setHorizontalSpacing(8)
        self.form.setVerticalSpacing(8)
        layout.addLayout(self.form)
        self.shape_rows = {}

        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.editingFinished.connect(self.load_input_mrc)
        input_button = QtWidgets.QPushButton("Browse")
        input_button.clicked.connect(self.choose_input_mrc)
        input_row = QtWidgets.QWidget()
        input_layout = QtWidgets.QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(6)
        input_layout.addWidget(self.input_edit, stretch=1)
        input_layout.addWidget(input_button)
        self.form.addRow("Input MRC", input_row)

        self.box_size_label = QtWidgets.QLabel("No file loaded")
        self.box_size_angstrom_label = QtWidgets.QLabel("No file loaded")
        self.pixel_size_label = QtWidgets.QLabel("No file loaded")
        self.form.addRow("Box size", self.box_size_label)
        self.form.addRow("Box size (A)", self.box_size_angstrom_label)
        self.form.addRow("Pixel size", self.pixel_size_label)

        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["sphere", "square-cube", "cylinder", "rectangle-cube"])
        self.shape_combo.currentTextChanged.connect(self._update_shape_controls)
        self.form.addRow("Shape", self.shape_combo)

        self.sphere_radius_spin = self._make_length_spinbox()
        self.sphere_z_offset_spin = self._make_signed_length_spinbox()
        self.cylinder_radius_spin = self._make_length_spinbox()
        self.cylinder_height_spin = self._make_length_spinbox()
        self.cube_x_spin = self._make_length_spinbox()
        self.cube_y_spin = self._make_length_spinbox()
        self.cube_z_spin = self._make_length_spinbox()
        self.orientation_combo = QtWidgets.QComboBox()
        self.orientation_combo.addItem("x (height along x)", "x")
        self.orientation_combo.addItem("y (height along y)", "y")
        self.orientation_combo.addItem("z (height along z)", "z")
        self.orientation_combo.setCurrentIndex(2)

        self._add_shape_row("sphere_radius", "Sphere radius", self.sphere_radius_spin)
        self._add_shape_row("sphere_z_offset", "Center Z offset", self.sphere_z_offset_spin)
        self._add_shape_row("cylinder_radius", "Cylinder radius", self.cylinder_radius_spin)
        self._add_shape_row("cylinder_height", "Cylinder height", self.cylinder_height_spin)
        self._add_shape_row("cube_x", "Cube X", self.cube_x_spin)
        self._add_shape_row("cube_y", "Cube Y", self.cube_y_spin)
        self._add_shape_row("cube_z", "Cube Z", self.cube_z_spin)
        self._add_shape_row("orientation", "Cylinder axis", self.orientation_combo)

        self.mask_output_edit = QtWidgets.QLineEdit()
        mask_button = QtWidgets.QPushButton("Browse")
        mask_button.clicked.connect(lambda: self.choose_output_path(self.mask_output_edit, "Save mask"))
        mask_row = QtWidgets.QWidget()
        mask_layout = QtWidgets.QHBoxLayout(mask_row)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.setSpacing(6)
        mask_layout.addWidget(self.mask_output_edit, stretch=1)
        mask_layout.addWidget(mask_button)
        self.form.addRow("Mask output", mask_row)

        self.masked_output_edit = QtWidgets.QLineEdit()
        masked_button = QtWidgets.QPushButton("Browse")
        masked_button.clicked.connect(lambda: self.choose_output_path(self.masked_output_edit, "Save masked MRC"))
        masked_row = QtWidgets.QWidget()
        masked_layout = QtWidgets.QHBoxLayout(masked_row)
        masked_layout.setContentsMargins(0, 0, 0, 0)
        masked_layout.setSpacing(6)
        masked_layout.addWidget(self.masked_output_edit, stretch=1)
        masked_layout.addWidget(masked_button)
        self.form.addRow("Masked output", masked_row)

        self.auto_load_checkbox = QtWidgets.QCheckBox("Auto-load masked output on generate?")
        self.auto_load_checkbox.setChecked(True)
        self.form.addRow("", self.auto_load_checkbox)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(120)
        layout.addWidget(self.status_box)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        generate_button = QtWidgets.QPushButton("Generate Mask")
        generate_button.clicked.connect(self.generate_outputs)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(generate_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _make_length_spinbox(self):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setSuffix(" A")
        return spin

    def _make_signed_length_spinbox(self):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setSuffix(" A")
        spin.setValue(0.0)
        return spin

    def _add_shape_row(self, key, label_text, field_widget):
        label = QtWidgets.QLabel(label_text)
        self.form.addRow(label, field_widget)
        self.shape_rows[key] = (label, field_widget)

    def choose_input_mrc(self):
        start_dir = str(Path(self.input_edit.text()).parent) if self.input_edit.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose MRC file",
            start_dir,
            "MRC files (*.mrc *.map *.mrcs);;All files (*)",
        )
        if not path:
            return
        self.input_edit.setText(path)
        self.load_input_mrc()

    def choose_output_path(self, line_edit, title):
        start_dir = str(Path(line_edit.text()).parent) if line_edit.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            start_dir,
            "MRC files (*.mrc);;All files (*)",
        )
        if path:
            line_edit.setText(path)

    def load_input_mrc(self):
        path_text = self.input_edit.text().strip()
        if not path_text:
            return
        input_path = Path(path_text)
        if not input_path.is_file():
            QtWidgets.QMessageBox.critical(self, "Load Failed", f"File not found:\n{input_path}")
            return

        try:
            self.volume, self.angstrom_per_pixel = load_mrc(input_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))
            return

        shape = tuple(int(value) for value in self.volume.shape)
        box_angstrom = np.array(self.volume.shape[::-1], dtype=np.float32) * float(self.angstrom_per_pixel)
        self.box_size_label.setText(f"{shape[2]} x {shape[1]} x {shape[0]} px")
        self.box_size_angstrom_label.setText(
            f"{box_angstrom[0]:.2f} x {box_angstrom[1]:.2f} x {box_angstrom[2]:.2f} A"
        )
        self.pixel_size_label.setText(f"{self.angstrom_per_pixel:.4f} A/px")
        self._set_default_dimensions()
        self._set_default_outputs(input_path)
        self.log(
            f"Loaded {input_path.name}: shape={shape}, pixel_size={self.angstrom_per_pixel:.4f} A/px"
        )

    def _set_default_dimensions(self):
        if self.volume is None:
            return
        box_angstrom = np.array(self.volume.shape[::-1], dtype=np.float32) * float(self.angstrom_per_pixel)
        min_extent = float(np.min(box_angstrom))
        self.sphere_radius_spin.setValue(round(min_extent * 0.25, 2))
        self.sphere_z_offset_spin.setValue(0.0)
        self.cube_x_spin.setValue(round(min_extent, 2))
        self.cylinder_radius_spin.setValue(round(min_extent * 0.20, 2))
        self.cylinder_height_spin.setValue(round(min_extent, 2))
        self.cube_y_spin.setValue(round(min_extent, 2))
        self.cube_z_spin.setValue(round(min_extent, 2))
        self._apply_launch_defaults()

    def _apply_launch_defaults(self):
        if not self.defaults:
            return
        shape_name = self.defaults.get("shape")
        if shape_name in {"sphere", "square-cube", "cylinder", "rectangle-cube"}:
            self.shape_combo.setCurrentText(shape_name)
        if "sphere_radius" in self.defaults:
            self.sphere_radius_spin.setValue(float(self.defaults["sphere_radius"]))
        if "sphere_z_offset" in self.defaults:
            self.sphere_z_offset_spin.setValue(float(self.defaults["sphere_z_offset"]))

    def _set_default_outputs(self, input_path):
        shape_name = self.shape_combo.currentText()
        mask_path = input_path.with_name(f"{input_path.stem}_{shape_name}_mask.mrc")
        masked_path = input_path.with_name(f"{input_path.stem}_{shape_name}_masked.mrc")
        self.mask_output_edit.setText(str(mask_path))
        self.masked_output_edit.setText(str(masked_path))

    def _update_shape_controls(self):
        shape_name = self.shape_combo.currentText()
        visible_rows = {
            "sphere": {"sphere_radius", "sphere_z_offset"},
            "square-cube": {"cube_x"},
            "cylinder": {"cylinder_radius", "cylinder_height", "orientation"},
            "rectangle-cube": {"cube_x", "cube_y", "cube_z"},
        }[shape_name]
        self.shape_rows["cube_x"][0].setText("Edge length" if shape_name == "square-cube" else "Length X")
        self.shape_rows["cube_y"][0].setText("Length Y")
        self.shape_rows["cube_z"][0].setText("Length Z")
        for key, widgets in self.shape_rows.items():
            visible = key in visible_rows
            for widget in widgets:
                widget.setVisible(visible)
        path_text = self.input_edit.text().strip()
        if path_text:
            self._set_default_outputs(Path(path_text))

    def _validate(self):
        if self.volume is None:
            raise ValueError("Load an input MRC first.")

        shape_name = self.shape_combo.currentText()
        dimensions = {
            "sphere_radius": float(self.sphere_radius_spin.value()),
            "sphere_z_offset": float(self.sphere_z_offset_spin.value()),
            "cylinder_radius": float(self.cylinder_radius_spin.value()),
            "cylinder_height": float(self.cylinder_height_spin.value()),
            "cube_x": float(self.cube_x_spin.value()),
            "cube_y": float(self.cube_y_spin.value()),
            "cube_z": float(self.cube_z_spin.value()),
        }
        required = {
            "sphere": ["sphere_radius"],
            "square-cube": ["cube_x"],
            "cylinder": ["cylinder_radius", "cylinder_height"],
            "rectangle-cube": ["cube_x", "cube_y", "cube_z"],
        }[shape_name]
        for key in required:
            if dimensions[key] <= 0.0:
                raise ValueError(f"{key.replace('_', ' ')} must be greater than zero.")

        mask_text = self.mask_output_edit.text().strip()
        masked_text = self.masked_output_edit.text().strip()
        if not mask_text:
            raise ValueError("Choose a mask output path.")
        if not masked_text:
            raise ValueError("Choose a masked output path.")
        mask_path = Path(mask_text)
        masked_path = Path(masked_text)
        return shape_name, dimensions, mask_path, masked_path

    def generate_outputs(self):
        try:
            shape_name, dimensions, mask_path, masked_path = self._validate()
            orientation = self.orientation_combo.currentData()
            mask = build_mask(
                self.volume.shape,
                self.angstrom_per_pixel,
                shape_name,
                dimensions,
                orientation,
            )
            masked_volume = self.volume.astype(np.float32) * mask
            write_mrc(mask_path, mask, self.angstrom_per_pixel)
            write_mrc(masked_path, masked_volume, self.angstrom_per_pixel)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Mask Generation Failed", str(exc))
            return

        self.log(f"Saved mask: {mask_path}")
        self.log(f"Saved masked map: {masked_path}")
        if self.auto_load_checkbox.isChecked():
            sys.stdout.write(
                json.dumps(
                    {
                        "type": "load_masked_output",
                        "path": str(masked_path),
                        "mask_summary": mask_summary(shape_name, dimensions, orientation),
                    }
                ) + "\n"
            )
            sys.stdout.flush()
        QtWidgets.QMessageBox.information(self, "Mask Generated", "Mask and masked MRC saved.")

    def log(self, text):
        self.status_box.appendPlainText(text)


def main():
    app = QtWidgets.QApplication(sys.argv)
    input_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    defaults = {}
    if len(sys.argv) > 2:
        try:
            defaults = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            defaults = {}
    window = MrcMaskHelper(input_path=input_path, defaults=defaults)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
