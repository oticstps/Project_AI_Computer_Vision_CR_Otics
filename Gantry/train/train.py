


# wand rahwana

import sys
import re
import time
import threading
from queue import Queue, Empty
from pathlib import Path

import serial
import serial.tools.list_ports

from PySide6.QtCore import QEvent, Qt, QThread, QObject, Signal
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def clean_gcode(text: str) -> list[str]:
    """Menghapus komentar dan baris kosong sebelum dikirim ke GRBL."""
    cleaned: list[str] = []

    for raw_line in text.splitlines():
        line = re.sub(r"\([^)]*\)", "", raw_line)
        line = line.split(";", 1)[0].strip()

        if not line or line == "%":
            continue

        cleaned.append(line)

    return cleaned


class GRBLSerialThread(QThread):
    """
    Thread serial khusus GRBL.

    Streaming menggunakan mekanisme satu baris -> tunggu 'ok'/'error'
    sehingga lebih aman daripada mengirim seluruh file dengan time.sleep().
    """

    log_signal = Signal(str)
    connected_signal = Signal(bool, str)
    machine_state_signal = Signal(str)
    position_signal = Signal(float, float, float, str)
    feed_spindle_signal = Signal(float, float)
    progress_signal = Signal(int, int)
    stream_finished_signal = Signal(bool, str)

    def __init__(self, port: str, baudrate: int = 115200):
        super().__init__()

        self.port = port
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None

        self._stop_event = threading.Event()
        self._normal_queue: Queue[str] = Queue()
        self._realtime_queue: Queue[bytes] = Queue()

        self._stream_lock = threading.Lock()
        self._stream_lines: list[str] = []
        self._stream_index = 0
        self._streaming = False
        self._stream_paused = False
        self._waiting_ack = False
        self._abort_requested = False

    @property
    def streaming(self) -> bool:
        with self._stream_lock:
            return self._streaming

    def queue_command(self, command: str) -> bool:
        command = command.strip()

        if not command:
            return False

        if self.streaming:
            self.log_signal.emit(
                "Perintah manual diblokir selama program G-code berjalan."
            )
            return False

        self._normal_queue.put(command)
        return True

    def queue_realtime(self, data: bytes) -> None:
        self._realtime_queue.put(data)

    def start_stream(self, lines: list[str]) -> bool:
        if not lines:
            self.log_signal.emit("Tidak ada G-code yang dapat dijalankan.")
            return False

        with self._stream_lock:
            if self._streaming:
                self.log_signal.emit("Program lain masih berjalan.")
                return False

            self._stream_lines = list(lines)
            self._stream_index = 0
            self._streaming = True
            self._stream_paused = False
            self._waiting_ack = False
            self._abort_requested = False

        self.progress_signal.emit(0, len(lines))
        self.log_signal.emit(f"Streaming dimulai: {len(lines)} baris.")
        return True

    def pause_stream(self) -> None:
        with self._stream_lock:
            if not self._streaming:
                return
            self._stream_paused = True

        self.queue_realtime(b"!")
        self.log_signal.emit("Program dijeda (feed hold).")

    def resume_stream(self) -> None:
        with self._stream_lock:
            if not self._streaming:
                return
            self._stream_paused = False

        self.queue_realtime(b"~")
        self.log_signal.emit("Program dilanjutkan.")

    def abort_stream(self) -> None:
        with self._stream_lock:
            if self._streaming:
                self._abort_requested = True
            else:
                self.queue_realtime(b"\x18")

    def request_stop(self) -> None:
        self._stop_event.set()

    def _write_line(self, command: str) -> None:
        if self.ser and self.ser.is_open:
            self.ser.write((command + "\n").encode("ascii", errors="ignore"))
            self.log_signal.emit(f">> {command}")

    def _write_realtime(self, data: bytes) -> None:
        if self.ser and self.ser.is_open:
            self.ser.write(data)

    def _clear_normal_queue(self) -> None:
        while True:
            try:
                self._normal_queue.get_nowait()
            except Empty:
                break

    def _finish_stream(self, success: bool, message: str) -> None:
        with self._stream_lock:
            self._streaming = False
            self._stream_paused = False
            self._waiting_ack = False
            self._abort_requested = False

        self.stream_finished_signal.emit(success, message)

    def _process_received_line(self, line: str) -> None:
        if not line:
            return

        # Status report GRBL: <Idle|MPos:...|FS:...>
        if line.startswith("<") and line.endswith(">"):
            self._parse_status_report(line)
            return

        self.log_signal.emit(f"<< {line}")

        lower = line.lower()

        if lower == "ok":
            with self._stream_lock:
                if not (self._streaming and self._waiting_ack):
                    return

                self._stream_index += 1
                self._waiting_ack = False
                current = self._stream_index
                total = len(self._stream_lines)

            self.progress_signal.emit(current, total)

            if current >= total:
                self._finish_stream(True, "Program selesai tanpa error.")

        elif lower.startswith("error"):
            with self._stream_lock:
                active = self._streaming
                line_number = self._stream_index + 1

            if active:
                self._finish_stream(
                    False,
                    f"Program dihentikan pada baris {line_number}: {line}",
                )

        elif lower.startswith("alarm"):
            with self._stream_lock:
                active = self._streaming

            if active:
                self._finish_stream(False, f"GRBL masuk kondisi alarm: {line}")

    def _parse_status_report(self, line: str) -> None:
        payload = line[1:-1]
        parts = payload.split("|")

        if not parts:
            return

        machine_state = parts[0]
        self.machine_state_signal.emit(machine_state)

        fields: dict[str, str] = {}

        for part in parts[1:]:
            if ":" in part:
                key, value = part.split(":", 1)
                fields[key] = value

        # Prioritaskan WPos agar DRO cocok dengan titik nol kerja.
        coordinate_key = "WPos" if "WPos" in fields else "MPos"

        if coordinate_key in fields:
            try:
                xyz = [float(value) for value in fields[coordinate_key].split(",")[:3]]

                if len(xyz) == 3:
                    self.position_signal.emit(
                        xyz[0],
                        xyz[1],
                        xyz[2],
                        coordinate_key,
                    )
            except (TypeError, ValueError):
                pass

        if "FS" in fields:
            try:
                fs = fields["FS"].split(",")
                feed = float(fs[0])
                spindle = float(fs[1]) if len(fs) > 1 else 0.0
                self.feed_spindle_signal.emit(feed, spindle)
            except (TypeError, ValueError):
                pass

    def run(self) -> None:
        last_status_query = 0.0

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.05,
                write_timeout=0.5,
            )

            # Banyak board GRBL melakukan reset ketika port serial dibuka.
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            self.connected_signal.emit(True, self.port)
            self.log_signal.emit(
                f"Terhubung ke {self.port} @ {self.baudrate} baud."
            )

            # Wake-up sequence, tanpa otomatis meng-unlock mesin.
            self.ser.write(b"\r\n\r\n")

            while not self._stop_event.is_set():
                now = time.monotonic()

                # Perintah realtime mempunyai prioritas tertinggi.
                while True:
                    try:
                        realtime_data = self._realtime_queue.get_nowait()
                    except Empty:
                        break
                    self._write_realtime(realtime_data)

                # Abort menggunakan soft reset GRBL.
                with self._stream_lock:
                    abort_requested = self._abort_requested

                if abort_requested:
                    self._write_realtime(b"!")
                    time.sleep(0.05)
                    self._write_realtime(b"\x18")
                    self._clear_normal_queue()
                    self._finish_stream(False, "Program dibatalkan dengan soft reset.")

                # Polling status.
                if now - last_status_query >= 0.25:
                    self._write_realtime(b"?")
                    last_status_query = now

                # Kirim satu baris G-code dan tunggu respons ok/error.
                with self._stream_lock:
                    streaming = self._streaming
                    paused = self._stream_paused
                    waiting_ack = self._waiting_ack
                    index = self._stream_index
                    total = len(self._stream_lines)

                if streaming and not paused and not waiting_ack and index < total:
                    command = self._stream_lines[index]
                    self._write_line(command)

                    with self._stream_lock:
                        self._waiting_ack = True

                elif not streaming:
                    try:
                        command = self._normal_queue.get_nowait()
                    except Empty:
                        command = None

                    if command:
                        self._write_line(command)

                try:
                    raw = self.ser.readline()
                except serial.SerialException as exc:
                    raise RuntimeError(f"Koneksi serial terputus: {exc}") from exc

                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    self._process_received_line(line)

                self.msleep(2)

        except Exception as exc:
            self.log_signal.emit(f"ERROR: {exc}")

        finally:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass

            self.connected_signal.emit(False, self.port)
            self.log_signal.emit("Koneksi serial ditutup.")


class GRBLController(QObject):
    log_signal = Signal(str)
    connected_signal = Signal(bool, str)
    machine_state_signal = Signal(str)
    position_signal = Signal(float, float, float, str)
    feed_spindle_signal = Signal(float, float)
    progress_signal = Signal(int, int)
    stream_finished_signal = Signal(bool, str)

    def __init__(self):
        super().__init__()

        self.worker: GRBLSerialThread | None = None
        self.connected = False

    @staticmethod
    def available_ports() -> list[str]:
        return [port.device for port in serial.tools.list_ports.comports()]

    @property
    def streaming(self) -> bool:
        return bool(self.worker and self.worker.streaming)

    def connect_serial(self, port: str, baudrate: int = 115200) -> None:
        if self.worker and self.worker.isRunning():
            self.log_signal.emit("Koneksi serial masih aktif.")
            return

        self.worker = GRBLSerialThread(port, baudrate)

        self.worker.log_signal.connect(self.log_signal)
        self.worker.connected_signal.connect(self._on_connection_changed)
        self.worker.machine_state_signal.connect(self.machine_state_signal)
        self.worker.position_signal.connect(self.position_signal)
        self.worker.feed_spindle_signal.connect(self.feed_spindle_signal)
        self.worker.progress_signal.connect(self.progress_signal)
        self.worker.stream_finished_signal.connect(self.stream_finished_signal)

        self.worker.start()

    def _on_connection_changed(self, connected: bool, port: str) -> None:
        self.connected = connected
        self.connected_signal.emit(connected, port)

    def disconnect_serial(self) -> None:
        if not self.worker:
            return

        self.worker.request_stop()
        self.worker.wait(1500)

        if self.worker.isRunning():
            self.log_signal.emit(
                "Thread serial belum berhenti sempurna; port akan ditutup saat loop berakhir."
            )

        self.connected = False

    def send(self, command: str) -> bool:
        if not self.connected or not self.worker:
            self.log_signal.emit("Belum terhubung ke GRBL.")
            return False

        return self.worker.queue_command(command)

    def realtime(self, data: bytes) -> None:
        if self.connected and self.worker:
            self.worker.queue_realtime(data)

    def jog(self, axis: str, distance: float, feed: int) -> bool:
        axis = axis.upper()

        if axis not in {"X", "Y", "Z"}:
            self.log_signal.emit(f"Axis tidak valid: {axis}")
            return False

        return self.send(f"$J=G91 {axis}{distance:.4f} F{feed}")

    def start_continuous_jog(
        self,
        axis: str,
        direction: int,
        feed: int,
    ) -> bool:
        """Mulai jog panjang; dihentikan dengan realtime Jog Cancel 0x85."""
        direction = 1 if direction >= 0 else -1

        # Jarak dibuat sangat panjang agar gerak berlangsung selama tombol
        # keyboard ditahan. GRBL tetap menghormati soft limit bila diaktifkan.
        continuous_distance = 100000.0 * direction
        return self.jog(axis, continuous_distance, feed)

    def cancel_jog(self) -> None:
        """Hentikan gerakan $J tanpa melakukan soft reset controller."""
        self.realtime(b"\x85")

    def home(self) -> None:
        self.send("$H")

    def unlock(self) -> None:
        self.send("$X")

    def zero_axis(self, axis: str) -> None:
        axis = axis.upper()

        if axis in {"X", "Y", "Z"}:
            self.send(f"G10 L20 P1 {axis}0")

    def zero_all(self) -> None:
        self.send("G10 L20 P1 X0 Y0 Z0")

    def spindle_on(self, rpm: int) -> None:
        self.send(f"M3 S{rpm}")

    def spindle_off(self) -> None:
        self.send("M5")

    def coolant_on(self) -> None:
        self.send("M8")

    def coolant_off(self) -> None:
        self.send("M9")

    def start_program(self, lines: list[str]) -> bool:
        if not self.connected or not self.worker:
            self.log_signal.emit("Belum terhubung ke GRBL.")
            return False

        return self.worker.start_stream(lines)

    def pause_program(self) -> None:
        if self.worker:
            self.worker.pause_stream()

    def resume_program(self) -> None:
        if self.worker:
            self.worker.resume_stream()

    def abort_program(self) -> None:
        if self.worker:
            self.worker.abort_stream()

    def soft_reset(self) -> None:
        if self.worker:
            self.worker.abort_stream()


class CNC_HMI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GRBL CNC Professional HMI — Light UI")
        self.resize(1450, 900)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.grbl = GRBLController()
        self.gcode_lines: list[str] = []
        self.loaded_file: Path | None = None

        self._keyboard_jog_key: int | None = None
        self._keyboard_jog_axis: str | None = None
        self._keyboard_jog_direction = 0
        self._default_keyboard_hint = (
            "Keyboard HOLD: ←/A dan →/D = X | ↑/W dan ↓/S = Y | "
            "PageUp/E dan PageDown/Q = Z | lepas tombol = STOP"
        )

        self._build_ui()
        self._connect_signals()
        self.refresh_ports()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f4f7fb;
                color: #1f2937;
                font-size: 14px;
            }

            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
                background: #f4f7fb;
                border: none;
            }

            QGroupBox {
                background: #ffffff;
                border: 1px solid #c7d3df;
                border-radius: 10px;
                margin-top: 14px;
                padding: 12px;
                font-weight: 700;
                color: #155e8a;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 7px;
                background: #f4f7fb;
            }

            QPushButton {
                background: #e8eef5;
                color: #17324d;
                border: 1px solid #b7c5d3;
                border-radius: 8px;
                padding: 10px 12px;
                min-height: 20px;
                font-weight: 650;
            }

            QPushButton:hover {
                background: #dbeafe;
                border-color: #60a5fa;
                color: #0f3e68;
            }

            QPushButton:pressed {
                background: #bfdbfe;
                border-color: #3b82f6;
            }

            QPushButton:focus {
                border: 2px solid #3b82f6;
            }

            QPushButton:disabled {
                background: #edf1f5;
                color: #94a3b8;
                border-color: #d8e0e8;
            }

            QComboBox, QSpinBox, QLineEdit {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #aebdca;
                border-radius: 7px;
                padding: 8px;
                selection-background-color: #bfdbfe;
                selection-color: #172033;
            }

            QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
                border: 2px solid #3b82f6;
            }

            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #aebdca;
                selection-background-color: #dbeafe;
                selection-color: #0f3e68;
            }

            QPlainTextEdit {
                background: #ffffff;
                color: #155e3b;
                border: 1px solid #b8c7d4;
                border-radius: 8px;
                padding: 6px;
                font-family: Consolas;
                font-size: 13px;
                selection-background-color: #bfdbfe;
                selection-color: #172033;
            }

            QProgressBar {
                border: 1px solid #aebdca;
                border-radius: 7px;
                text-align: center;
                min-height: 22px;
                background: #e5ebf1;
                color: #17324d;
                font-weight: 700;
            }

            QProgressBar::chunk {
                background: #4f9bd8;
                border-radius: 6px;
            }

            QTabWidget::pane {
                background: #ffffff;
                border: 1px solid #c7d3df;
                border-radius: 8px;
                top: -1px;
            }

            QTabBar::tab {
                background: #e7edf4;
                color: #334155;
                border: 1px solid #c7d3df;
                border-bottom: none;
                padding: 10px 18px;
                margin-right: 2px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }

            QTabBar::tab:hover {
                background: #dbeafe;
            }

            QTabBar::tab:selected {
                background: #ffffff;
                color: #155e8a;
                font-weight: 800;
            }

            QSplitter::handle {
                background: #d4dee8;
                width: 5px;
            }

            QLabel#TitleLabel {
                color: #12395b;
                font-size: 23px;
                font-weight: 800;
            }

            QLabel#DROLabel {
                background: #f8fbfd;
                border: 1px solid #9fb6c8;
                border-radius: 8px;
                padding: 10px;
                color: #075985;
                font-family: Consolas;
                font-size: 27px;
                font-weight: 750;
            }

            QLabel#WarningLabel {
                color: #9a3412;
                background: #fff7ed;
                border: 1px solid #fdba74;
                border-radius: 8px;
                padding: 9px;
            }

            QToolTip {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #94a3b8;
                padding: 5px;
            }

            QScrollBar:vertical {
                background: #edf2f7;
                width: 12px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #b7c5d3;
                min-height: 30px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background: #94a8ba;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # Header
        header = QHBoxLayout()

        title = QLabel("⚙ GRBL CNC PROFESSIONAL HMI — LIGHT UI")
        title.setObjectName("TitleLabel")
        header.addWidget(title)
        header.addStretch()

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(130)
        header.addWidget(self.port_combo)

        self.refresh_button = QPushButton("REFRESH PORT")
        self.refresh_button.setToolTip("Memindai ulang port serial")
        header.addWidget(self.refresh_button)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["115200", "57600", "38400", "9600"])
        header.addWidget(self.baud_combo)

        self.connect_button = QPushButton("CONNECT")
        self.connect_button.setMinimumWidth(120)
        header.addWidget(self.connect_button)

        self.connection_badge = QLabel("DISCONNECTED")
        self.connection_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_badge.setMinimumWidth(150)
        self._set_connection_badge(False)
        header.addWidget(self.connection_badge)

        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        # LEFT PANEL
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 8, 2)

        dro_group = QGroupBox("DIGITAL READOUT")
        dro_layout = QVBoxLayout(dro_group)

        self.coordinate_type_label = QLabel("Coordinate: -")
        dro_layout.addWidget(self.coordinate_type_label)

        self.x_label = QLabel("X  +000.000")
        self.y_label = QLabel("Y  +000.000")
        self.z_label = QLabel("Z  +000.000")

        for label in (self.x_label, self.y_label, self.z_label):
            label.setObjectName("DROLabel")
            dro_layout.addWidget(label)

        info_row = QHBoxLayout()
        self.feed_actual_label = QLabel("F: 0")
        self.spindle_actual_label = QLabel("S: 0")
        info_row.addWidget(self.feed_actual_label)
        info_row.addWidget(self.spindle_actual_label)
        dro_layout.addLayout(info_row)

        left_layout.addWidget(dro_group)

        jog_group = QGroupBox("MANUAL JOG")
        jog_layout = QVBoxLayout(jog_group)

        setting_row = QHBoxLayout()

        setting_row.addWidget(QLabel("Step"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.01", "0.1", "1", "10", "50"])
        self.step_combo.setCurrentText("1")
        setting_row.addWidget(self.step_combo)

        setting_row.addWidget(QLabel("Feed"))
        self.jog_feed_spin = QSpinBox()
        self.jog_feed_spin.setRange(1, 10000)
        self.jog_feed_spin.setValue(1000)
        self.jog_feed_spin.setSuffix(" mm/min")
        setting_row.addWidget(self.jog_feed_spin)

        jog_layout.addLayout(setting_row)

        xy_grid = QGridLayout()

        self.y_plus_button = QPushButton("▲\nY+")
        self.x_minus_button = QPushButton("◀ X−")
        self.x_plus_button = QPushButton("X+ ▶")
        self.y_minus_button = QPushButton("Y−\n▼")

        xy_grid.addWidget(self.y_plus_button, 0, 1)
        xy_grid.addWidget(self.x_minus_button, 1, 0)
        xy_grid.addWidget(QLabel("XY", alignment=Qt.AlignmentFlag.AlignCenter), 1, 1)
        xy_grid.addWidget(self.x_plus_button, 1, 2)
        xy_grid.addWidget(self.y_minus_button, 2, 1)

        jog_layout.addLayout(xy_grid)

        z_row = QHBoxLayout()
        self.z_plus_button = QPushButton("Z+  PAGE UP")
        self.z_minus_button = QPushButton("Z−  PAGE DOWN")
        z_row.addWidget(self.z_plus_button)
        z_row.addWidget(self.z_minus_button)
        jog_layout.addLayout(z_row)

        left_layout.addWidget(jog_group)

        machine_group = QGroupBox("MACHINE CONTROL")
        machine_grid = QGridLayout(machine_group)

        self.home_button = QPushButton("HOME  ($H)")
        self.unlock_button = QPushButton("UNLOCK  ($X)")
        self.reset_button = QPushButton("SOFT RESET")
        self.feed_hold_button = QPushButton("FEED HOLD")
        self.cycle_start_button = QPushButton("CYCLE START")

        machine_grid.addWidget(self.home_button, 0, 0)
        machine_grid.addWidget(self.unlock_button, 0, 1)
        machine_grid.addWidget(self.feed_hold_button, 1, 0)
        machine_grid.addWidget(self.cycle_start_button, 1, 1)
        machine_grid.addWidget(self.reset_button, 2, 0, 1, 2)

        left_layout.addWidget(machine_group)

        zero_group = QGroupBox("SET WORK ZERO")
        zero_grid = QGridLayout(zero_group)

        self.zero_x_button = QPushButton("ZERO X")
        self.zero_y_button = QPushButton("ZERO Y")
        self.zero_z_button = QPushButton("ZERO Z")
        self.zero_all_button = QPushButton("ZERO XYZ")

        zero_grid.addWidget(self.zero_x_button, 0, 0)
        zero_grid.addWidget(self.zero_y_button, 0, 1)
        zero_grid.addWidget(self.zero_z_button, 1, 0)
        zero_grid.addWidget(self.zero_all_button, 1, 1)

        left_layout.addWidget(zero_group)

        spindle_group = QGroupBox("SPINDLE & COOLANT")
        spindle_layout = QGridLayout(spindle_group)

        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(0, 50000)
        self.rpm_spin.setValue(10000)
        self.rpm_spin.setSuffix(" RPM")

        self.spindle_on_button = QPushButton("SPINDLE ON")
        self.spindle_off_button = QPushButton("SPINDLE OFF")
        self.coolant_on_button = QPushButton("COOLANT ON")
        self.coolant_off_button = QPushButton("COOLANT OFF")

        spindle_layout.addWidget(self.rpm_spin, 0, 0, 1, 2)
        spindle_layout.addWidget(self.spindle_on_button, 1, 0)
        spindle_layout.addWidget(self.spindle_off_button, 1, 1)
        spindle_layout.addWidget(self.coolant_on_button, 2, 0)
        spindle_layout.addWidget(self.coolant_off_button, 2, 1)

        left_layout.addWidget(spindle_group)

        warning = QLabel(
            "Tombol software bukan pengganti Emergency Stop fisik. "
            "Gunakan E-Stop hardware yang memutus daya aktuator/spindle."
        )
        warning.setWordWrap(True)
        warning.setObjectName("WarningLabel")
        left_layout.addWidget(warning)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setMinimumWidth(410)
        splitter.addWidget(left_scroll)

        # RIGHT PANEL
        tabs = QTabWidget()
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Program tab
        program_tab = QWidget()
        program_layout = QVBoxLayout(program_tab)

        file_row = QHBoxLayout()
        self.file_label = QLabel("Belum ada file G-code")
        self.load_button = QPushButton("LOAD G-CODE")
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.load_button)
        program_layout.addLayout(file_row)

        self.gcode_preview = QPlainTextEdit()
        self.gcode_preview.setReadOnly(True)
        self.gcode_preview.setPlaceholderText(
            "Preview G-code akan tampil di sini..."
        )
        program_layout.addWidget(self.gcode_preview, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        program_layout.addWidget(self.progress_bar)

        program_button_row = QHBoxLayout()

        self.start_button = QPushButton("▶ START PROGRAM")
        self.pause_button = QPushButton("⏸ PAUSE")
        self.resume_button = QPushButton("⏵ RESUME")
        self.abort_button = QPushButton("■ ABORT / SOFT RESET")

        self.abort_button.setStyleSheet(
            "background:#fee2e2; color:#991b1b; border:1px solid #ef4444;"
        )

        program_button_row.addWidget(self.start_button)
        program_button_row.addWidget(self.pause_button)
        program_button_row.addWidget(self.resume_button)
        program_button_row.addWidget(self.abort_button)

        program_layout.addLayout(program_button_row)
        tabs.addTab(program_tab, "PROGRAM")

        # Console tab
        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.document().setMaximumBlockCount(3000)
        console_layout.addWidget(self.console, 1)

        command_row = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText(
            "Ketik perintah GRBL/G-code, contoh: $$ atau G0 X10"
        )
        self.send_button = QPushButton("SEND")
        self.clear_console_button = QPushButton("CLEAR")

        command_row.addWidget(self.command_input, 1)
        command_row.addWidget(self.send_button)
        command_row.addWidget(self.clear_console_button)
        console_layout.addLayout(command_row)

        tabs.addTab(console_tab, "CONSOLE")

        # Footer status
        footer = QHBoxLayout()

        footer.addWidget(QLabel("Machine state:"))
        self.machine_state_label = QLabel("UNKNOWN")
        self.machine_state_label.setStyleSheet(
            "background:#e2e8f0; color:#334155; border:1px solid #94a3b8; "
            "padding:6px 14px; border-radius:8px; font-weight:700;"
        )
        footer.addWidget(self.machine_state_label)
        footer.addStretch()

        self.keyboard_hint = QLabel(self._default_keyboard_hint)
        footer.addWidget(self.keyboard_hint)

        root_layout.addLayout(footer)

        self._set_program_running(False)

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.toggle_connection)

        self.grbl.log_signal.connect(self.log)
        self.grbl.connected_signal.connect(self.on_connection_changed)
        self.grbl.machine_state_signal.connect(self.update_machine_state)
        self.grbl.position_signal.connect(self.update_position)
        self.grbl.feed_spindle_signal.connect(self.update_feed_spindle)
        self.grbl.progress_signal.connect(self.update_progress)
        self.grbl.stream_finished_signal.connect(self.on_stream_finished)

        self.x_minus_button.clicked.connect(lambda: self.jog("X", -1))
        self.x_plus_button.clicked.connect(lambda: self.jog("X", 1))
        self.y_minus_button.clicked.connect(lambda: self.jog("Y", -1))
        self.y_plus_button.clicked.connect(lambda: self.jog("Y", 1))
        self.z_minus_button.clicked.connect(lambda: self.jog("Z", -1))
        self.z_plus_button.clicked.connect(lambda: self.jog("Z", 1))

        self.home_button.clicked.connect(self.confirm_home)
        self.unlock_button.clicked.connect(self.grbl.unlock)
        self.feed_hold_button.clicked.connect(lambda: self.grbl.realtime(b"!"))
        self.cycle_start_button.clicked.connect(lambda: self.grbl.realtime(b"~"))
        self.reset_button.clicked.connect(self.confirm_soft_reset)

        self.zero_x_button.clicked.connect(lambda: self.confirm_zero("X"))
        self.zero_y_button.clicked.connect(lambda: self.confirm_zero("Y"))
        self.zero_z_button.clicked.connect(lambda: self.confirm_zero("Z"))
        self.zero_all_button.clicked.connect(lambda: self.confirm_zero("ALL"))

        self.spindle_on_button.clicked.connect(
            lambda: self.grbl.spindle_on(self.rpm_spin.value())
        )
        self.spindle_off_button.clicked.connect(self.grbl.spindle_off)
        self.coolant_on_button.clicked.connect(self.grbl.coolant_on)
        self.coolant_off_button.clicked.connect(self.grbl.coolant_off)

        self.load_button.clicked.connect(self.load_gcode_file)
        self.start_button.clicked.connect(self.start_program)
        self.pause_button.clicked.connect(self.pause_program)
        self.resume_button.clicked.connect(self.resume_program)
        self.abort_button.clicked.connect(self.abort_program)

        self.send_button.clicked.connect(self.send_manual_command)
        self.command_input.returnPressed.connect(self.send_manual_command)
        self.clear_console_button.clicked.connect(self.console.clear)

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        ports = self.grbl.available_ports()

        self.port_combo.clear()
        self.port_combo.addItems(ports)

        if current in ports:
            self.port_combo.setCurrentText(current)
        elif "COM5" in ports:
            self.port_combo.setCurrentText("COM5")

        if not ports:
            self.port_combo.addItem("NO PORT")
            self.log("Tidak ditemukan port serial.")

    def toggle_connection(self) -> None:
        if self.grbl.connected:
            self.grbl.disconnect_serial()
            return

        port = self.port_combo.currentText().strip()

        if not port or port == "NO PORT":
            QMessageBox.warning(
                self,
                "Port tidak tersedia",
                "Hubungkan controller GRBL lalu tekan REFRESH PORT.",
            )
            return

        baudrate = int(self.baud_combo.currentText())
        self.connect_button.setEnabled(False)
        self.connection_badge.setText("CONNECTING...")
        self.grbl.connect_serial(port, baudrate)

    def on_connection_changed(self, connected: bool, port: str) -> None:
        if not connected:
            self._stop_keyboard_jog(send_cancel=False)

        self._set_connection_badge(connected)
        self.connect_button.setEnabled(True)
        self.connect_button.setText("DISCONNECT" if connected else "CONNECT")

        self.port_combo.setEnabled(not connected)
        self.baud_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)

        if not connected:
            self._set_program_running(False)

    def _set_connection_badge(self, connected: bool) -> None:
        if connected:
            self.connection_badge.setText("CONNECTED")
            self.connection_badge.setStyleSheet(
                "background:#dcfce7; color:#166534; border:1px solid #22c55e;"
                "border-radius:8px; padding:8px; font-weight:800;"
            )
        else:
            self.connection_badge.setText("DISCONNECTED")
            self.connection_badge.setStyleSheet(
                "background:#fee2e2; color:#991b1b; border:1px solid #ef4444;"
                "border-radius:8px; padding:8px; font-weight:800;"
            )

    def update_machine_state(self, state: str) -> None:
        self.machine_state_label.setText(state)

        state_upper = state.upper()

        if state_upper.startswith("IDLE"):
            background, foreground, border = "#dcfce7", "#166534", "#22c55e"
        elif state_upper.startswith("RUN"):
            background, foreground, border = "#dbeafe", "#1d4ed8", "#60a5fa"
        elif state_upper.startswith("HOLD"):
            background, foreground, border = "#fef3c7", "#92400e", "#f59e0b"
        elif state_upper.startswith(("ALARM", "DOOR")):
            background, foreground, border = "#fee2e2", "#991b1b", "#ef4444"
        else:
            background, foreground, border = "#e2e8f0", "#334155", "#94a3b8"

        self.machine_state_label.setStyleSheet(
            f"background:{background}; color:{foreground}; border:1px solid {border}; "
            "padding:6px 14px; border-radius:8px; font-weight:700;"
        )

    def update_position(
        self,
        x: float,
        y: float,
        z: float,
        coordinate_type: str,
    ) -> None:
        self.x_label.setText(f"X  {x:+010.3f}")
        self.y_label.setText(f"Y  {y:+010.3f}")
        self.z_label.setText(f"Z  {z:+010.3f}")
        self.coordinate_type_label.setText(f"Coordinate: {coordinate_type}")

    def update_feed_spindle(self, feed: float, spindle: float) -> None:
        self.feed_actual_label.setText(f"F: {feed:.0f}")
        self.spindle_actual_label.setText(f"S: {spindle:.0f}")

    def jog(self, axis: str, direction: int) -> None:
        if self.grbl.streaming:
            self.log("Jog diblokir selama program berjalan.")
            return

        step = float(self.step_combo.currentText())
        distance = step * direction
        self.grbl.jog(axis, distance, self.jog_feed_spin.value())

    def confirm_home(self) -> None:
        answer = QMessageBox.question(
            self,
            "Homing mesin",
            "Pastikan area gerak aman dan limit switch berfungsi.\n\n"
            "Jalankan homing cycle sekarang?",
        )

        if answer == QMessageBox.StandardButton.Yes:
            self.grbl.home()

    def confirm_soft_reset(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Soft reset GRBL",
            "Soft reset akan menghentikan gerakan dan dapat menghapus status modal/"
            "posisi internal tertentu.\n\nLanjutkan?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer == QMessageBox.StandardButton.Yes:
            self.grbl.soft_reset()

    def confirm_zero(self, axis: str) -> None:
        label = "X, Y, dan Z" if axis == "ALL" else axis

        answer = QMessageBox.question(
            self,
            "Set work zero",
            f"Set posisi kerja {label} saat ini menjadi nol?",
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        if axis == "ALL":
            self.grbl.zero_all()
        else:
            self.grbl.zero_axis(axis)

    def load_gcode_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open G-code",
            "",
            "G-code (*.nc *.gcode *.tap *.txt);;All files (*.*)",
        )

        if not filename:
            return

        path = Path(filename)

        try:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")

            lines = clean_gcode(text)

            if not lines:
                raise ValueError("File tidak berisi perintah G-code yang valid.")

            self.gcode_lines = lines
            self.loaded_file = path

            preview_limit = 1500
            preview_lines = lines[:preview_limit]
            preview = "\n".join(
                f"{index + 1:05d}  {line}"
                for index, line in enumerate(preview_lines)
            )

            if len(lines) > preview_limit:
                preview += (
                    f"\n\n... preview dibatasi {preview_limit} dari "
                    f"{len(lines)} baris ..."
                )

            self.gcode_preview.setPlainText(preview)
            self.file_label.setText(f"{path.name} — {len(lines)} baris")
            self.progress_bar.setRange(0, len(lines))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0 / {len(lines)}")
            self.log(f"Loaded: {path} ({len(lines)} baris bersih)")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Gagal membuka file",
                str(exc),
            )

    def start_program(self) -> None:
        self._stop_keyboard_jog()

        if not self.grbl.connected:
            QMessageBox.warning(
                self,
                "Belum terhubung",
                "Hubungkan aplikasi ke controller GRBL terlebih dahulu.",
            )
            return

        if not self.gcode_lines:
            QMessageBox.warning(
                self,
                "G-code kosong",
                "Pilih file G-code terlebih dahulu.",
            )
            return

        answer = QMessageBox.warning(
            self,
            "Konfirmasi menjalankan program",
            "Pastikan:\n"
            "• toolpath telah diverifikasi;\n"
            "• benda kerja terjepit kuat;\n"
            "• work zero sudah benar;\n"
            "• spindle, tool, dan area gerak aman;\n"
            "• E-Stop fisik dapat dijangkau.\n\n"
            "Mulai program?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        if self.grbl.start_program(self.gcode_lines):
            self._set_program_running(True)

    def pause_program(self) -> None:
        self.grbl.pause_program()
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)

    def resume_program(self) -> None:
        self.grbl.resume_program()
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)

    def abort_program(self) -> None:
        if not self.grbl.streaming:
            return

        answer = QMessageBox.warning(
            self,
            "Batalkan program",
            "Gerakan akan dihentikan menggunakan feed hold lalu soft reset.\n\n"
            "Batalkan program sekarang?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer == QMessageBox.StandardButton.Yes:
            self.grbl.abort_program()

    def update_progress(self, current: int, total: int) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current} / {total}")

    def on_stream_finished(self, success: bool, message: str) -> None:
        self._set_program_running(False)
        self.log(message)

        if success:
            QMessageBox.information(self, "Program selesai", message)
        else:
            QMessageBox.warning(self, "Program dihentikan", message)

    def _set_program_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.resume_button.setEnabled(False)
        self.abort_button.setEnabled(running)

        self.home_button.setEnabled(not running)
        self.unlock_button.setEnabled(not running)
        self.zero_x_button.setEnabled(not running)
        self.zero_y_button.setEnabled(not running)
        self.zero_z_button.setEnabled(not running)
        self.zero_all_button.setEnabled(not running)
        self.spindle_on_button.setEnabled(not running)
        self.spindle_off_button.setEnabled(not running)
        self.coolant_on_button.setEnabled(not running)
        self.coolant_off_button.setEnabled(not running)

    def send_manual_command(self) -> None:
        command = self.command_input.text().strip()

        if not command:
            return

        if self.grbl.send(command):
            self.command_input.clear()

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{timestamp}] {message}")

    def _keyboard_jog_blocked(self) -> bool:
        """Mencegah jog saat pengguna sedang mengetik/mengubah nilai input."""
        focused = QApplication.focusWidget()

        if isinstance(focused, (QLineEdit, QComboBox, QSpinBox)):
            return True

        if isinstance(focused, QPlainTextEdit) and not focused.isReadOnly():
            return True

        return QApplication.activeModalWidget() is not None

    @staticmethod
    def _keyboard_movement(key: int) -> tuple[str, int] | None:
        movement_map = {
            Qt.Key.Key_Left: ("X", -1),
            Qt.Key.Key_A: ("X", -1),
            Qt.Key.Key_Right: ("X", 1),
            Qt.Key.Key_D: ("X", 1),
            Qt.Key.Key_Up: ("Y", 1),
            Qt.Key.Key_W: ("Y", 1),
            Qt.Key.Key_Down: ("Y", -1),
            Qt.Key.Key_S: ("Y", -1),
            Qt.Key.Key_PageUp: ("Z", 1),
            Qt.Key.Key_E: ("Z", 1),
            Qt.Key.Key_PageDown: ("Z", -1),
            Qt.Key.Key_Q: ("Z", -1),
        }
        return movement_map.get(key)

    def _start_keyboard_jog(
        self,
        key: int,
        axis: str,
        direction: int,
    ) -> None:
        if self.grbl.streaming:
            self.log("Jog keyboard diblokir selama program berjalan.")
            return

        if not self.grbl.connected:
            self.log("Belum terhubung ke GRBL.")
            return

        # Hanya satu arah keyboard aktif pada satu waktu. Bila arah diganti
        # sebelum tombol lama dilepas, batalkan jog lama terlebih dahulu.
        if self._keyboard_jog_key is not None:
            if self._keyboard_jog_key == key:
                return
            self._stop_keyboard_jog()

        if self.grbl.start_continuous_jog(
            axis,
            direction,
            self.jog_feed_spin.value(),
        ):
            self._keyboard_jog_key = key
            self._keyboard_jog_axis = axis
            self._keyboard_jog_direction = direction
            sign = "+" if direction > 0 else "−"
            self.keyboard_hint.setText(
                f"HOLD JOG AKTIF: {axis}{sign} — lepas tombol untuk berhenti"
            )
            self.keyboard_hint.setStyleSheet(
                "background:#fef3c7; color:#92400e; border:1px solid #f59e0b; "
                "padding:6px 10px; border-radius:7px; font-weight:700;"
            )

    def _stop_keyboard_jog(self, send_cancel: bool = True) -> None:
        if self._keyboard_jog_key is None:
            return

        if send_cancel and self.grbl.connected:
            # GRBL v1.1 realtime Jog Cancel. Tidak mereset posisi/modal state.
            self.grbl.cancel_jog()

        self._keyboard_jog_key = None
        self._keyboard_jog_axis = None
        self._keyboard_jog_direction = 0

        if hasattr(self, "keyboard_hint"):
            self.keyboard_hint.setText(self._default_keyboard_hint)
            self.keyboard_hint.setStyleSheet("")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Tekan-tahan untuk jog kontinu; lepas tombol untuk Jog Cancel."""
        event_type = event.type()

        # Hindari mesin terus bergerak bila aplikasi kehilangan fokus sebelum
        # menerima KeyRelease, misalnya pengguna Alt+Tab saat tombol ditahan.
        if event_type in {
            QEvent.Type.WindowDeactivate,
            QEvent.Type.ApplicationDeactivate,
        }:
            self._stop_keyboard_jog()
            return super().eventFilter(watched, event)

        if event_type not in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
            return super().eventFilter(watched, event)

        movement = self._keyboard_movement(event.key())

        if movement is None:
            return super().eventFilter(watched, event)

        # Event auto-repeat menghasilkan pasangan press/release semu. Abaikan
        # semuanya; jog dimulai dari press pertama dan berhenti pada release asli.
        if event.isAutoRepeat():
            event.accept()
            return True

        if event_type == QEvent.Type.KeyRelease:
            if self._keyboard_jog_key == event.key():
                self._stop_keyboard_jog()
                event.accept()
                return True
            return super().eventFilter(watched, event)

        if not self.isActiveWindow() or self._keyboard_jog_blocked():
            return super().eventFilter(watched, event)

        axis, direction = movement
        self._start_keyboard_jog(event.key(), axis, direction)
        event.accept()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_keyboard_jog()

        if self.grbl.streaming:
            answer = QMessageBox.warning(
                self,
                "Program masih berjalan",
                "Program G-code masih berjalan. Batalkan dan tutup aplikasi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.grbl.abort_program()
            time.sleep(0.1)

        self.grbl.disconnect_serial()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("GRBL CNC Professional HMI Light")

    window = CNC_HMI()
    app.installEventFilter(window)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
