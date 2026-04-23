import os
import json
import shlex
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".mplconfig"))
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from pyvistaqt import QtInteractor
except ImportError:
    QtInteractor = None

import volume_viewer_pyvista_chain_palettes_gif as chain_viewer
import volume_viewer_pyvista_ken10_experimental_gif as ken10_viewer

EMBED_PREVIEW_DEFAULT = sys.platform not in {"linux", "linux2"}


def labeled_groupbox(title, layout):
    box = QtWidgets.QGroupBox(title)
    box.setLayout(layout)
    return box


class PathField(QtWidgets.QWidget):
    def __init__(self, label, default_path="", save=False, parent=None):
        super().__init__(parent)
        self.save = save
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(default_path)
        self.edit.setPlaceholderText(label)
        self.button = QtWidgets.QPushButton("Browse")
        self.button.clicked.connect(self.browse)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def browse(self):
        if self.save:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Choose Output GIF",
                self.edit.text() or str(REPO_DIR / "output.gif"),
                "GIF files (*.gif);;All files (*)",
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose File",
                self.edit.text() or str(REPO_DIR),
                "All files (*)",
            )
        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()

    def setText(self, value):
        self.edit.setText(value)


class ViewerTab(QtWidgets.QWidget):
    def __init__(self, main_window, mode):
        super().__init__()
        self.main_window = main_window
        self.mode = mode
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(labeled_groupbox("Files", self._build_files_form()))
        root.addWidget(labeled_groupbox("Rendering", self._build_render_form()))
        root.addWidget(labeled_groupbox("Options", self._build_options_form()))
        root.addWidget(labeled_groupbox("Mode-Specific", self._build_mode_specific_form()))
        root.addStretch(1)
        self._connect_live_preview_signals()

    def _build_files_form(self):
        layout = QtWidgets.QFormLayout()
        self.input_file = PathField("Input MRC", str(self.default_input_file()))
        self.output_file = PathField("Output GIF", str(self.default_output_file()), save=True)
        layout.addRow("Input MRC", self.input_file)
        layout.addRow("Output GIF", self.output_file)

        if self.mode == "chain":
            self.cif_file = PathField("Chain mmCIF", str(REPO_DIR / "7M3N.cif"))
            layout.addRow("Chain mmCIF", self.cif_file)
        else:
            self.major_head_cif = PathField(
                "Major head mmCIF", str(REPO_DIR / "10u_mcp-job113_filtered.cif")
            )
            self.minor_head1_cif = PathField(
                "Minor head 1 mmCIF", str(REPO_DIR / "10v_minorHead1-job114-filtered.cif")
            )
            self.minor_head2_cif = PathField(
                "Minor head 2 mmCIF", str(REPO_DIR / "10w-minorHead2-job115_filtered.cif")
            )
            layout.addRow("Major head mmCIF", self.major_head_cif)
            layout.addRow("Minor head 1 mmCIF", self.minor_head1_cif)
            layout.addRow("Minor head 2 mmCIF", self.minor_head2_cif)

        return layout

    def _build_render_form(self):
        layout = QtWidgets.QFormLayout()
        self.angpix = self._double_spin(0.0, 100.0, 0.0, 4, 0.1, special="Auto")
        self.level = self._double_spin(0.0, 1_000_000.0, 0.0, 4, 0.01, special="Auto")
        self.step_size = self._spin(1, 16, 2)
        self.distance_cutoff = self._double_spin(0.1, 100.0, 4.0, 2, 0.1)
        self.elev = self._double_spin(-180.0, 180.0, 20.0, 1, 1.0)
        self.azim = self._double_spin(-360.0, 360.0, 0.0, 1, 1.0)
        self.symop = self._spin(1, 60, 1)
        self.cmap = QtWidgets.QComboBox()
        module = chain_viewer if self.mode == "chain" else ken10_viewer
        self.cmap.addItems(module.AVAILABLE_CMAPS)
        self.cmap.setCurrentText(module.DEFAULT_BASE_CMAP)
        self.background = QtWidgets.QLineEdit("white")
        self.quality = QtWidgets.QComboBox()
        self.quality.addItems(list(module.QUALITY_PRESETS.keys()))
        self.quality.setCurrentText("better")
        self.fps = self._spin(0, 120, 0, special="Auto")
        self.base_boxsize = QtWidgets.QComboBox()
        self.base_boxsize.addItems(["128", "256", "512", "1024"])
        self.base_boxsize.setCurrentText("512")
        self.crop_pad = self._spin(0, 500, 0)
        self.crop_size_mode = QtWidgets.QComboBox()
        self.crop_size_mode.addItems(["multiple_of_10", "power_of_2"])

        layout.addRow("Angpix", self.angpix)
        layout.addRow("Level", self.level)
        layout.addRow("Step size", self.step_size)
        layout.addRow("Distance cutoff", self.distance_cutoff)
        layout.addRow("Elevation", self.elev)
        layout.addRow("Azimuth", self.azim)
        layout.addRow("Symmetry op", self.symop)
        layout.addRow("Colormap", self.cmap)
        layout.addRow("Background", self.background)
        layout.addRow("Quality", self.quality)
        layout.addRow("FPS", self.fps)
        layout.addRow("Base box size", self.base_boxsize)
        layout.addRow("Crop pad", self.crop_pad)
        layout.addRow("Crop size mode", self.crop_size_mode)
        return layout

    def _build_options_form(self):
        layout = QtWidgets.QGridLayout()
        self.hide_dust = QtWidgets.QCheckBox("Hide dust")
        self.fast = QtWidgets.QCheckBox("Fast mode")
        self.crop = QtWidgets.QCheckBox("Crop output")
        self.scale_bar = QtWidgets.QCheckBox("Scale bar")
        self.dust_volume_cutoff = self._double_spin(0.0, 100_000.0, 4.0, 2, 0.5)
        layout.addWidget(self.hide_dust, 0, 0)
        layout.addWidget(self.fast, 0, 1)
        layout.addWidget(self.crop, 1, 0)
        layout.addWidget(self.scale_bar, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Dust volume cutoff"), 2, 0)
        layout.addWidget(self.dust_volume_cutoff, 2, 1)
        return layout

    def _build_mode_specific_form(self):
        layout = QtWidgets.QFormLayout()
        if self.mode == "chain":
            self.chain_palette = QtWidgets.QPlainTextEdit()
            self.chain_palette.setPlaceholderText("One spec per line, e.g. A:YlOrRd\nH,L:PuBuGn")
            self.symmetry_copy_chains = QtWidgets.QLineEdit()
            self.symmetry_copy_chains.setPlaceholderText("Comma-separated chains or semicolon-separated groups")
            layout.addRow("Chain palettes", self.chain_palette)
            layout.addRow("Symmetry copy chains", self.symmetry_copy_chains)
        else:
            self.total_rotation = self._double_spin(1.0, 3600.0, 360.0, 1, 5.0)
            self.speed_scale = self._double_spin(0.1, 10.0, 1.0, 2, 0.1)
            self.lighting_preset = QtWidgets.QComboBox()
            self.lighting_preset.addItems(list(ken10_viewer.LIGHTING_PRESETS))
            self.silhouette = QtWidgets.QCheckBox("Silhouette")
            self.outline = QtWidgets.QCheckBox("Outline")
            self.smooth_shading = QtWidgets.QCheckBox("Smooth shading")
            self.shadows = QtWidgets.QCheckBox("Shadows")
            self.silhouette_color = QtWidgets.QLineEdit("black")
            self.outline_color = QtWidgets.QLineEdit("black")
            self.silhouette_width = self._double_spin(0.0, 10.0, 0.0, 2, 0.1, special="Auto")
            self.outline_width = self._double_spin(0.1, 10.0, 1.5, 2, 0.1)
            layout.addRow("Total rotation", self.total_rotation)
            layout.addRow("Speed scale", self.speed_scale)
            layout.addRow("Lighting preset", self.lighting_preset)
            layout.addRow("Silhouette", self.silhouette)
            layout.addRow("Silhouette color", self.silhouette_color)
            layout.addRow("Silhouette width", self.silhouette_width)
            layout.addRow("Outline", self.outline)
            layout.addRow("Outline color", self.outline_color)
            layout.addRow("Outline width", self.outline_width)
            layout.addRow("Smooth shading", self.smooth_shading)
            layout.addRow("Shadows", self.shadows)
        return layout

    def _spin(self, minimum, maximum, value, special=None):
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        if special is not None:
            widget.setSpecialValueText(special)
        return widget

    def _double_spin(self, minimum, maximum, value, decimals, step, special=None):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        if special is not None:
            widget.setSpecialValueText(special)
        return widget

    def default_input_file(self):
        if self.mode == "chain":
            return REPO_DIR / "fab.mrc"
        return REPO_DIR / "KEN10.mrc"

    def default_output_file(self):
        if self.mode == "chain":
            return REPO_DIR / "output-fab-gui.gif"
        return REPO_DIR / "output-ken10-gui.gif"

    def reset_paths(self):
        self.input_file.setText(str(self.default_input_file()))
        self.output_file.setText(str(self.default_output_file()))
        if self.mode == "chain":
            self.cif_file.setText(str(REPO_DIR / "7M3N.cif"))
        else:
            self.major_head_cif.setText(str(REPO_DIR / "10u_mcp-job113_filtered.cif"))
            self.minor_head1_cif.setText(str(REPO_DIR / "10v_minorHead1-job114-filtered.cif"))
            self.minor_head2_cif.setText(str(REPO_DIR / "10w-minorHead2-job115_filtered.cif"))

    def _connect_live_preview_signals(self):
        widgets = [
            self.input_file.edit,
            self.output_file.edit,
            self.angpix,
            self.level,
            self.step_size,
            self.distance_cutoff,
            self.elev,
            self.azim,
            self.symop,
            self.cmap,
            self.background,
            self.quality,
            self.fps,
            self.base_boxsize,
            self.crop_pad,
            self.crop_size_mode,
            self.hide_dust,
            self.fast,
            self.crop,
            self.scale_bar,
            self.dust_volume_cutoff,
        ]

        if self.mode == "chain":
            widgets.extend(
                [
                    self.cif_file.edit,
                    self.chain_palette,
                    self.symmetry_copy_chains,
                ]
            )
        else:
            widgets.extend(
                [
                    self.major_head_cif.edit,
                    self.minor_head1_cif.edit,
                    self.minor_head2_cif.edit,
                    self.total_rotation,
                    self.speed_scale,
                    self.lighting_preset,
                    self.silhouette,
                    self.outline,
                    self.smooth_shading,
                    self.shadows,
                    self.silhouette_color,
                    self.outline_color,
                    self.silhouette_width,
                    self.outline_width,
                ]
            )

        for widget in widgets:
            self._connect_widget_signal(widget)

    def _connect_widget_signal(self, widget):
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit)):
            signal = widget.textChanged
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            signal = widget.valueChanged
        elif isinstance(widget, QtWidgets.QComboBox):
            signal = widget.currentTextChanged
        elif isinstance(widget, QtWidgets.QCheckBox):
            signal = widget.toggled
        else:
            return
        signal.connect(self.main_window.schedule_live_preview)

    def _optional_float(self, widget):
        if widget.value() == 0:
            return None
        return float(widget.value())

    def _optional_int(self, widget):
        if widget.value() == 0:
            return None
        return int(widget.value())

    def _chain_palette_specs(self):
        return [
            line.strip()
            for line in self.chain_palette.toPlainText().splitlines()
            if line.strip()
        ]

    def _symmetry_copy_specs(self):
        text = self.symmetry_copy_chains.text().strip()
        if not text:
            return []
        return [part.strip() for part in text.split(";") if part.strip()]

    def preview_config(self):
        config = {
            "input_file": self.input_file.text(),
            "angpix": self._optional_float(self.angpix),
            "level": self._optional_float(self.level),
            "step_size": int(self.step_size.value()),
            "distance_cutoff": float(self.distance_cutoff.value()),
            "hide_dust": self.hide_dust.isChecked(),
            "dust_volume_cutoff": float(self.dust_volume_cutoff.value()),
            "symop": int(self.symop.value()),
            "elev": float(self.elev.value()),
            "azim": float(self.azim.value()),
            "cmap": self.cmap.currentText(),
            "background": self.background.text().strip() or "white",
            "fast": self.fast.isChecked(),
        }
        if self.mode == "chain":
            config["cif_file"] = self.cif_file.text()
            config["chain_palette_specs"] = self._chain_palette_specs()
            config["symmetry_copy_specs"] = self._symmetry_copy_specs()
        else:
            config["major_head_cif"] = self.major_head_cif.text()
            config["minor_head1_cif"] = self.minor_head1_cif.text()
            config["minor_head2_cif"] = self.minor_head2_cif.text()
            config["lighting_preset"] = self.lighting_preset.currentText()
            config["silhouette"] = self.silhouette.isChecked()
            config["silhouette_color"] = self.silhouette_color.text().strip() or "black"
            config["silhouette_width"] = self._optional_float(self.silhouette_width)
            config["outline"] = self.outline.isChecked()
            config["outline_color"] = self.outline_color.text().strip() or "black"
            config["outline_width"] = float(self.outline_width.value())
            config["smooth_shading"] = self.smooth_shading.isChecked()
            config["shadows"] = self.shadows.isChecked()
        return config

    def export_args(self):
        args = [self.input_file.text(), self.output_file.text()]
        args.extend(self._common_cli_args())

        if self.mode == "chain":
            args.extend(["--cif-file", self.cif_file.text()])
            for spec in self._chain_palette_specs():
                args.extend(["--chain-palette", spec])
            for spec in self._symmetry_copy_specs():
                args.extend(["--symmetry-copy-chains", spec])
        else:
            args.extend(["--major-head-cif", self.major_head_cif.text()])
            args.extend(["--minor-head1-cif", self.minor_head1_cif.text()])
            args.extend(["--minor-head2-cif", self.minor_head2_cif.text()])
            args.extend(["--total-rotation", f"{self.total_rotation.value():.1f}"])
            args.extend(["--speed-scale", f"{self.speed_scale.value():.2f}"])
            args.extend(["--lighting-preset", self.lighting_preset.currentText()])
            if self.silhouette.isChecked():
                args.append("--silhouette")
                args.extend(["--silhouette-color", self.silhouette_color.text().strip() or "black"])
                if self.silhouette_width.value() > 0:
                    args.extend(["--silhouette-width", f"{self.silhouette_width.value():.2f}"])
            if self.outline.isChecked():
                args.append("--outline")
                args.extend(["--outline-color", self.outline_color.text().strip() or "black"])
                args.extend(["--outline-width", f"{self.outline_width.value():.2f}"])
            if self.smooth_shading.isChecked():
                args.append("--smooth-shading")
            if self.shadows.isChecked():
                args.append("--shadows")

        return args

    def _common_cli_args(self):
        args = [
            "--distance-cutoff",
            f"{self.distance_cutoff.value():.2f}",
            "--step-size",
            str(self.step_size.value()),
            "--elev",
            f"{self.elev.value():.1f}",
            "--azim",
            f"{self.azim.value():.1f}",
            "--cmap",
            self.cmap.currentText(),
            "--background",
            self.background.text().strip() or "white",
            "--quality",
            self.quality.currentText(),
            "--base-boxsize",
            self.base_boxsize.currentText(),
            "--crop-pad",
            str(self.crop_pad.value()),
            "--crop-size-mode",
            self.crop_size_mode.currentText(),
            "--dust-volume-cutoff",
            f"{self.dust_volume_cutoff.value():.2f}",
            "--symop",
            str(self.symop.value()),
        ]
        angpix = self._optional_float(self.angpix)
        level = self._optional_float(self.level)
        fps = self._optional_int(self.fps)
        if angpix is not None:
            args.extend(["--angpix", f"{angpix:.4f}"])
        if level is not None:
            args.extend(["--level", f"{level:.4f}"])
        if fps is not None:
            args.extend(["--fps", str(fps)])
        if self.hide_dust.isChecked():
            args.append("--hide-dust")
        if self.fast.isChecked():
            args.append("--fast")
        if self.crop.isChecked():
            args.append("--crop")
        if self.scale_bar.isChecked():
            args.append("--scale-bar")
        return args


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISECC Volume Viewer GUI")
        self.resize(1400, 820)
        self.process = None
        self.preview_process = None
        self.preview_output_buffer = ""
        self.preview_config_path = None
        self.current_mesh = None
        self.preview_generation = 0
        self.applied_preview_generation = 0
        self.pending_live_preview = False
        self.embed_preview_enabled = (
            QtInteractor is not None
            and os.environ.get("ISECC_ENABLE_EMBEDDED_PREVIEW", "").lower()
            in {"1", "true", "yes"}
        ) or (QtInteractor is not None and EMBED_PREVIEW_DEFAULT)
        self.live_preview_enabled = True
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.tabs = QtWidgets.QTabWidget()
        self.chain_tab = ViewerTab(self, "chain")
        self.ken10_tab = ViewerTab(self, "ken10")
        self.tabs.addTab(self.chain_tab, "Chain Palettes")
        self.tabs.addTab(self.ken10_tab, "KEN10 Experimental")
        left_layout.addWidget(self.tabs)

        button_row = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview Mesh")
        self.export_button = QtWidgets.QPushButton("Export GIF")
        self.reset_paths_button = QtWidgets.QPushButton("Reset Paths")
        self.stop_button = QtWidgets.QPushButton("Stop Export")
        self.live_preview_checkbox = QtWidgets.QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        self.stop_button.setEnabled(False)
        self.preview_button.clicked.connect(self.start_preview)
        self.export_button.clicked.connect(self.start_export)
        self.reset_paths_button.clicked.connect(self.reset_active_paths)
        self.stop_button.clicked.connect(self.stop_export)
        self.live_preview_checkbox.toggled.connect(self._set_live_preview_enabled)
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.live_preview_checkbox)
        button_row.addWidget(self.reset_paths_button)
        button_row.addWidget(self.stop_button)
        left_layout.addLayout(button_row)

        self.command_preview = QtWidgets.QLineEdit()
        self.command_preview.setReadOnly(True)
        left_layout.addWidget(self.command_preview)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        left_layout.addWidget(self.log, stretch=1)
        left_panel.setMinimumWidth(380)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        if self.embed_preview_enabled:
            self.plotter = QtInteractor(right_panel)
            self.plotter.set_background("white")
            self.plotter.add_axes()
            right_layout.addWidget(self.plotter, stretch=1)
        else:
            self.plotter = None
            self.preview_note = QtWidgets.QLabel(
                "Off-screen preview mode is active.\n\n"
                "Click Preview Mesh to render a single PNG snapshot here.\n"
                "This avoids the X11/VTK embedded viewport crash on this system.\n\n"
                "To force embedded preview back on, launch with:\n"
                "`ISECC_ENABLE_EMBEDDED_PREVIEW=1 python volume_viewer_gui.py`"
            )
            self.preview_note.setWordWrap(True)
            self.preview_note.setAlignment(QtCore.Qt.AlignCenter)
            self.preview_image = QtWidgets.QLabel()
            self.preview_image.setAlignment(QtCore.Qt.AlignCenter)
            self.preview_image.setMinimumSize(320, 240)
            self.preview_image.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self.preview_image.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.preview_image.setText("Preview image will appear here.")
            right_layout.addWidget(self.preview_note)
            right_layout.addWidget(self.preview_image, stretch=1)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 940])
        self.tabs.currentChanged.connect(self.refresh_command_preview)
        self.tabs.currentChanged.connect(self.schedule_live_preview)
        self.refresh_command_preview()

        self.live_preview_timer = QtCore.QTimer(self)
        self.live_preview_timer.setSingleShot(True)
        self.live_preview_timer.setInterval(700)
        self.live_preview_timer.timeout.connect(self._start_live_preview_if_possible)

    def active_tab(self):
        return self.tabs.currentWidget()

    def active_mode(self):
        return "chain" if self.tabs.currentWidget() is self.chain_tab else "ken10"

    def append_log(self, text):
        if not text:
            return
        self.log.appendPlainText(text.rstrip())

    def refresh_command_preview(self):
        tab = self.active_tab()
        script_name = (
            "volume_viewer_pyvista_chain_palettes_gif.py"
            if self.active_mode() == "chain"
            else "volume_viewer_pyvista_ken10_experimental_gif.py"
        )
        command = [sys.executable, script_name] + tab.export_args()
        self.command_preview.setText(" ".join(shlex.quote(part) for part in command))

    def _set_live_preview_enabled(self, enabled):
        self.live_preview_enabled = enabled
        if enabled:
            self.schedule_live_preview()
        else:
            self.live_preview_timer.stop()
            self.pending_live_preview = False

    def schedule_live_preview(self, *_args):
        self.refresh_command_preview()
        if not self.live_preview_enabled:
            return
        self.live_preview_timer.start()

    def _start_live_preview_if_possible(self):
        if self.preview_process is not None and self.preview_process.state() != QtCore.QProcess.NotRunning:
            self.pending_live_preview = True
            return
        self.start_preview(auto=True)

    def reset_active_paths(self):
        self.active_tab().reset_paths()
        if self.plotter is None:
            self.preview_image.clear()
            self.preview_image.setText("Preview image will appear here.")
            self.preview_image.setToolTip("")
        self.refresh_command_preview()
        self.append_log("Paths reset to default values.")
        self.schedule_live_preview()

    def _validate_main_inputs(self):
        tab = self.active_tab()
        input_path = Path(tab.input_file.text())
        if not input_path.is_file():
            raise FileNotFoundError(f"Input MRC not found: {input_path}")

        if self.active_mode() == "chain":
            cif_path = Path(tab.cif_file.text())
            if not cif_path.is_file():
                raise FileNotFoundError(f"Chain mmCIF not found: {cif_path}")
        else:
            required_paths = [
                ("Major head mmCIF", Path(tab.major_head_cif.text())),
                ("Minor head 1 mmCIF", Path(tab.minor_head1_cif.text())),
                ("Minor head 2 mmCIF", Path(tab.minor_head2_cif.text())),
            ]
            for label, path in required_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"{label} not found: {path}")

    def start_preview(self, auto=False):
        self.refresh_command_preview()
        if self.plotter is not None:
            self._fail_preview(
                "Embedded live preview is currently unsupported in this build.",
                show_dialog=not auto,
            )
            return

        if self.preview_process is not None and self.preview_process.state() != QtCore.QProcess.NotRunning:
            if auto:
                self.pending_live_preview = True
                return
            self.append_log("Preview already in progress.")
            return
        self.preview_button.setEnabled(False)
        if not auto:
            self.append_log("Building preview mesh...")
        try:
            self._validate_main_inputs()
        except Exception as exc:
            self._fail_preview(str(exc), show_dialog=not auto)
            return
        config = self.active_tab().preview_config()
        self.preview_generation += 1
        generation = self.preview_generation
        preview_dir = REPO_DIR / ".preview_cache"
        preview_dir.mkdir(exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="isecc-preview-config-",
            dir=preview_dir,
            delete=False,
        )
        handle.write(json.dumps(config).encode("utf-8"))
        handle.close()
        self.preview_config_path = handle.name

        self.preview_process = QtCore.QProcess(self)
        self.preview_output_buffer = ""
        self.preview_process.setProgram(sys.executable)
        self.preview_process.setArguments(
            [
                str(REPO_DIR / "volume_viewer_preview_snapshot.py"),
                self.active_mode(),
                self.preview_config_path,
            ]
        )
        self.preview_process.setWorkingDirectory(str(REPO_DIR))
        self.preview_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.preview_process.readyReadStandardOutput.connect(self._drain_preview_output)
        self.preview_process.finished.connect(
            lambda exit_code, exit_status, generation=generation, auto=auto: self._finish_preview_process(
                exit_code,
                exit_status,
                generation,
                auto,
            )
        )
        self.preview_process.start()

    def _drain_preview_output(self):
        if self.preview_process is None:
            return ""
        data = bytes(self.preview_process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        if data:
            self.preview_output_buffer += data
            self.append_log(data)
        return data

    def _finish_preview_process(self, exit_code, exit_status, generation, auto):
        self._drain_preview_output()
        output = self.preview_output_buffer
        image_path = None
        for line in output.splitlines():
            if line.startswith("PREVIEW_IMAGE="):
                image_path = line.split("=", 1)[1].strip()

        if self.preview_config_path:
            try:
                Path(self.preview_config_path).unlink()
            except OSError:
                pass
            self.preview_config_path = None

        if generation >= self.applied_preview_generation and exit_status == QtCore.QProcess.NormalExit and exit_code == 0 and image_path:
            self.applied_preview_generation = generation
            pixmap = QtGui.QPixmap(image_path)
            if pixmap.isNull():
                self.preview_image.setText(f"Preview rendered but could not load image:\n{image_path}")
            else:
                scaled = pixmap.scaled(
                    self.preview_image.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.preview_image.setPixmap(scaled)
            self.preview_image.setToolTip(image_path)
            self.preview_button.setEnabled(True)
            self.append_log("Preview image ready.")
        else:
            message = f"Preview process failed with exit code {exit_code}."
            if exit_status != QtCore.QProcess.NormalExit:
                message = "Preview process crashed."
            self._fail_preview(message, show_dialog=not auto, generation=generation)

        self.preview_process.deleteLater()
        self.preview_process = None
        self.preview_output_buffer = ""
        if self.pending_live_preview and self.live_preview_enabled:
            self.pending_live_preview = False
            self.live_preview_timer.start(50)

    def _fail_preview(self, message, show_dialog=True, generation=None):
        if generation is not None and generation < self.applied_preview_generation:
            self.preview_button.setEnabled(True)
            return
        self.preview_button.setEnabled(True)
        if self.plotter is None:
            self.preview_image.clear()
            self.preview_image.setText(
                "Preview unavailable.\n\n"
                f"{message}"
            )
            self.preview_image.setToolTip("")
        self.append_log(f"Preview failed: {message}")
        if show_dialog:
            QtWidgets.QMessageBox.critical(self, "Preview Failed", message)

    def start_export(self):
        self.refresh_command_preview()
        if self.process is not None:
            return
        try:
            self._validate_main_inputs()
        except Exception as exc:
            self.append_log(f"Export blocked: {exc}")
            QtWidgets.QMessageBox.critical(self, "Export Blocked", str(exc))
            return

        tab = self.active_tab()
        script = (
            REPO_DIR / "volume_viewer_pyvista_chain_palettes_gif.py"
            if self.active_mode() == "chain"
            else REPO_DIR / "volume_viewer_pyvista_ken10_experimental_gif.py"
        )
        args = [str(script)] + tab.export_args()

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.setWorkingDirectory(str(REPO_DIR))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._drain_process_output)
        self.process.finished.connect(self._finish_export)
        self.process.start()

        self.export_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.append_log(f"Started export: {script.name}")

    def _drain_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(data)

    def _finish_export(self, exit_code, exit_status):
        self._drain_process_output()
        success = exit_status == QtCore.QProcess.NormalExit and exit_code == 0
        message = "Export finished successfully." if success else f"Export failed with exit code {exit_code}."
        self.append_log(message)
        self.export_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process.deleteLater()
        self.process = None

    def stop_export(self):
        if self.process is None:
            return
        self.process.kill()
        self.append_log("Export stopped.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ISECC Volume Viewer GUI")
    app.setWindowIcon(QtGui.QIcon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
