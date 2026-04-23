import os
import sys
from pathlib import Path

import mrcfile
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6 import QtCore, QtWidgets
from skimage import measure

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
    "cubehelix",
    "Spectral",
    "coolwarm",
)


def load_volume(mrc_path):
    with mrcfile.open(mrc_path) as mrc:
        volume = np.array(mrc.data, copy=True)
    volume = np.flipud(volume)
    volume = iseccFFT_v3.swapAxes_ndimage(volume)
    return volume


def default_threshold(volume):
    return float(np.around((np.amax(volume) / 10.0) + 0.004, decimals=4))


def centered_vertices(verts, shape):
    center = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    return verts - center


def downsample_volume_to_target(volume, max_display_size):
    max_dim = int(max(volume.shape))
    if max_dim <= max_display_size:
        return volume, {"factor": 1, "shape": volume.shape}

    factor = int(np.ceil(max_dim / float(max_display_size)))
    factor = max(1, factor)
    downsampled = volume[::factor, ::factor, ::factor]
    return downsampled, {"factor": factor, "shape": downsampled.shape}


class MeshWorker(QtCore.QObject):
    finished = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(
        self,
        generation,
        volume,
        threshold,
        step_size,
        max_display_size,
    ):
        super().__init__()
        self.generation = generation
        self.volume = volume
        self.threshold = threshold
        self.step_size = step_size
        self.max_display_size = max_display_size

    @QtCore.Slot()
    def run(self):
        try:
            working_volume, downsample_info = downsample_volume_to_target(
                self.volume,
                self.max_display_size,
            )
            verts, faces, _, _ = measure.marching_cubes(
                working_volume,
                level=self.threshold,
                step_size=self.step_size,
            )
            verts = centered_vertices(verts, working_volume.shape)
            triangles = verts[faces]
            centroids = triangles.mean(axis=1)
            face_values = np.linalg.norm(centroids, axis=1)
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))
            return

        payload = {
            "verts": verts,
            "faces": faces,
            "face_values": face_values,
            "threshold": self.threshold,
            "downsample_info": downsample_info,
        }
        self.finished.emit(self.generation, payload)


class IsosurfaceCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        figure = Figure(figsize=(7, 7), tight_layout=True)
        super().__init__(figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.collection = None
        self.mesh_payload = None
        self.current_palette = PALETTES[0]
        self.current_azim = 0.0
        self.current_elev = 25.0
        self._draw_empty("Open an MRC file to begin.")

    def _draw_empty(self, text):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_axis_off()
        self.axes.text2D(
            0.5,
            0.5,
            text,
            transform=self.axes.transAxes,
            ha="center",
            va="center",
        )
        self.draw_idle()

    def set_mesh(self, payload, palette):
        self.mesh_payload = payload
        self.current_palette = palette
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_facecolor("white")
        self.axes.set_axis_off()

        verts = payload["verts"]
        faces = payload["faces"]
        triangles = verts[faces]
        facecolors = self._facecolors(payload["face_values"], palette)

        self.collection = Poly3DCollection(
            triangles,
            facecolors=facecolors,
            linewidths=0.0,
            edgecolors="none",
            alpha=1.0,
        )
        self.axes.add_collection3d(self.collection)
        self._set_limits(verts)
        self.axes.view_init(elev=self.current_elev, azim=self.current_azim)
        self.draw_idle()

    def recolor(self, palette):
        self.current_palette = palette
        if self.mesh_payload is None or self.collection is None:
            return
        self.collection.set_facecolors(
            self._facecolors(self.mesh_payload["face_values"], palette)
        )
        self.draw_idle()

    def rotate_to(self, azim):
        self.current_azim = azim % 360.0
        if self.mesh_payload is None:
            return
        self.axes.view_init(elev=self.current_elev, azim=self.current_azim)
        self.draw_idle()

    def _facecolors(self, values, palette):
        cmap = colormaps.get_cmap(palette)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            normalized = np.zeros_like(values, dtype=np.float64)
        else:
            normalized = (values - vmin) / (vmax - vmin)
        return cmap(normalized)

    def _set_limits(self, verts):
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = np.max(maxs - mins) / 2.0
        radius = max(radius, 1.0)

        self.axes.set_xlim(center[0] - radius, center[0] + radius)
        self.axes.set_ylim(center[1] - radius, center[1] + radius)
        self.axes.set_zlim(center[2] - radius, center[2] + radius)
        self.axes.set_box_aspect((1, 1, 1))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRC Isosurface Rotation Viewer")
        self.resize(1280, 820)

        self.volume = None
        self.current_file = None
        self.mesh_thread = None
        self.mesh_worker = None
        self.mesh_generation = 0
        self.applied_generation = 0
        self.pending_remesh = False
        self.rotation_angle = 0.0

        self._build_ui()
        self._build_timers()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container)

        file_group = QtWidgets.QGroupBox("File")
        file_layout = QtWidgets.QFormLayout(file_group)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("Choose an MRC file")
        self.file_edit.returnPressed.connect(self.load_selected_mrc)
        open_button = QtWidgets.QPushButton("Open MRC")
        open_button.clicked.connect(self.choose_mrc)
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(self.file_edit)
        file_row.addWidget(open_button)
        file_row_widget = QtWidgets.QWidget()
        file_row_widget.setLayout(file_row)
        file_layout.addRow("MRC path", file_row_widget)
        controls_layout.addWidget(file_group)

        render_group = QtWidgets.QGroupBox("Surface")
        render_layout = QtWidgets.QFormLayout(render_group)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.threshold_spin.valueChanged.connect(self.schedule_remesh)

        self.palette_combo = QtWidgets.QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.currentTextChanged.connect(self.on_palette_changed)

        self.step_size_spin = QtWidgets.QSpinBox()
        self.step_size_spin.setRange(1, 8)
        self.step_size_spin.setValue(2)
        self.step_size_spin.valueChanged.connect(self.schedule_remesh)

        self.display_size_combo = QtWidgets.QComboBox()
        self.display_size_combo.addItems(["96", "128", "160", "192", "256", "320", "384", "512"])
        self.display_size_combo.setCurrentText("160")
        self.display_size_combo.currentTextChanged.connect(self.schedule_remesh)

        render_layout.addRow("Density threshold", self.threshold_spin)
        render_layout.addRow("Color palette", self.palette_combo)
        render_layout.addRow("Mesh step size", self.step_size_spin)
        render_layout.addRow("Display volume max size", self.display_size_combo)
        controls_layout.addWidget(render_group)

        motion_group = QtWidgets.QGroupBox("Rotation")
        motion_layout = QtWidgets.QFormLayout(motion_group)
        self.rotate_checkbox = QtWidgets.QCheckBox("Rotate continuously")
        self.rotate_checkbox.setChecked(True)
        self.rotate_checkbox.toggled.connect(self.toggle_rotation)

        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 20.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setValue(2.0)
        self.speed_spin.setSuffix(" deg/frame")

        motion_layout.addRow(self.rotate_checkbox)
        motion_layout.addRow("Rotation speed", self.speed_spin)
        controls_layout.addWidget(motion_group)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumBlockCount(200)
        controls_layout.addWidget(self.status_box, stretch=1)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_container)
        controls_scroll.setMinimumWidth(340)

        viewer_panel = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
        self.canvas = IsosurfaceCanvas()
        viewer_layout.addWidget(self.canvas)

        splitter.addWidget(controls_scroll)
        splitter.addWidget(viewer_panel)
        splitter.setSizes([360, 920])

    def _build_timers(self):
        self.remesh_timer = QtCore.QTimer(self)
        self.remesh_timer.setSingleShot(True)
        self.remesh_timer.setInterval(450)
        self.remesh_timer.timeout.connect(self.start_remesh)

        self.rotation_timer = QtCore.QTimer(self)
        self.rotation_timer.setInterval(40)
        self.rotation_timer.timeout.connect(self.advance_rotation)
        self.rotation_timer.start()

    def choose_mrc(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose MRC file",
            self.file_edit.text() or str(REPO_DIR),
            "MRC files (*.mrc *.map *.mrcs);;All files (*)",
        )
        if not path:
            return
        self.file_edit.setText(path)
        self.load_selected_mrc()

    def load_selected_mrc(self):
        path = self.file_edit.text().strip()
        if not path:
            return
        mrc_path = Path(path)
        if not mrc_path.is_file():
            self.log(f"File not found: {mrc_path}")
            QtWidgets.QMessageBox.critical(self, "Open Failed", f"File not found:\n{mrc_path}")
            return

        try:
            self.volume = load_volume(mrc_path)
        except Exception as exc:
            self.log(f"Failed to load volume: {exc}")
            QtWidgets.QMessageBox.critical(self, "Open Failed", str(exc))
            return

        self.current_file = mrc_path
        threshold = default_threshold(self.volume)
        vol_min = float(np.min(self.volume))
        vol_max = float(np.max(self.volume))
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setRange(vol_min, vol_max)
        self.threshold_spin.setValue(min(max(threshold, vol_min), vol_max))
        self.threshold_spin.blockSignals(False)

        self.log(
            f"Loaded {mrc_path.name}: shape={self.volume.shape}, "
            f"min={vol_min:.4f}, max={vol_max:.4f}"
        )
        self.schedule_remesh()

    def schedule_remesh(self, *_args):
        if self.volume is None:
            return
        self.remesh_timer.start()

    def start_remesh(self):
        if self.volume is None:
            return
        if self.mesh_thread is not None and self.mesh_thread.isRunning():
            self.pending_remesh = True
            return

        threshold = float(self.threshold_spin.value())
        step_size = int(self.step_size_spin.value())
        self.mesh_generation += 1
        generation = self.mesh_generation
        self.log(
            f"Building mesh: threshold={threshold:.4f}, step_size={step_size}"
        )

        self.mesh_thread = QtCore.QThread(self)
        self.mesh_worker = MeshWorker(
            generation,
            self.volume,
            threshold,
            step_size,
            int(self.display_size_combo.currentText()),
        )
        self.mesh_worker.moveToThread(self.mesh_thread)
        self.mesh_thread.started.connect(self.mesh_worker.run)
        self.mesh_worker.finished.connect(self.on_mesh_ready)
        self.mesh_worker.failed.connect(self.on_mesh_failed)
        self.mesh_worker.finished.connect(self.mesh_thread.quit)
        self.mesh_worker.failed.connect(self.mesh_thread.quit)
        self.mesh_thread.finished.connect(self.mesh_worker.deleteLater)
        self.mesh_thread.finished.connect(self.mesh_thread.deleteLater)
        self.mesh_thread.finished.connect(self.on_mesh_thread_finished)
        self.mesh_thread.start()

    def on_mesh_ready(self, generation, payload):
        if generation < self.applied_generation:
            return
        self.applied_generation = generation
        self.canvas.set_mesh(payload, self.palette_combo.currentText())
        verts = payload["verts"]
        faces = payload["faces"]
        self.log(
            f"Mesh ready: verts={len(verts)}, faces={len(faces)}, "
            f"threshold={payload['threshold']:.4f}"
        )
        downsample_info = payload.get("downsample_info")
        if downsample_info is not None:
            self.log(
                "Downsample applied: "
                f"factor={int(downsample_info['factor'])}, "
                f"shape={tuple(int(x) for x in downsample_info['shape'])}"
            )

    def on_mesh_failed(self, generation, message):
        if generation >= self.applied_generation:
            self.log(f"Mesh build failed: {message}")
            QtWidgets.QMessageBox.critical(self, "Mesh Build Failed", message)

    def on_mesh_thread_finished(self):
        self.mesh_worker = None
        self.mesh_thread = None
        if self.pending_remesh:
            self.pending_remesh = False
            self.remesh_timer.start(50)

    def on_palette_changed(self, palette):
        self.canvas.recolor(palette)

    def toggle_rotation(self, enabled):
        if enabled:
            self.rotation_timer.start()
        else:
            self.rotation_timer.stop()

    def advance_rotation(self):
        if not self.rotate_checkbox.isChecked():
            return
        self.rotation_angle = (self.rotation_angle + float(self.speed_spin.value())) % 360.0
        self.canvas.rotate_to(self.rotation_angle)

    def log(self, text):
        self.status_box.appendPlainText(text)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
