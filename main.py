import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os

class DinoLiteYOLODetector:
    def __init__(self, model_path):
        """
        Inisialisasi detektor YOLOv8 untuk Dino-Lite
        
        Args:
            model_path: Path ke model YOLOv8 (.pt)
        """
        self.model_path = model_path
        self.cap = None
        self.model = None
        self.window_name = "Dino-Lite YOLOv8 Detection"
        
        # Konfigurasi Dino-Lite
        self.camera_index = 0  # Coba 0, 1, 2, atau 3 untuk Dino-Lite
        self.is_dinolite = False
        
        # Pengaturan tampilan
        self.fullscreen = False
        self.zoom_factor = 1.0
        self.pan_x, self.pan_y = 0, 0
        
        # Pengaturan pencahayaan Dino-Lite
        self.brightness = 50
        self.contrast = 50
        self.exposure = -1  # Auto exposure default
        
        # Pengaturan deteksi
        self.confidence_threshold = 0.10
        self.iou_threshold = 0.45
        
        # Recording
        self.recording = False
        self.video_writer = None
        self.record_fps = 30
        
        # Statistik
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self.prev_time = 0
        
        # Warna untuk class yang berbeda
        self.colors = [
            (0, 255, 0),    # Hijau
            (255, 0, 0),    # Biru
            (0, 0, 255),    # Merah
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Kuning
            (128, 0, 128),  # Ungu
            (128, 128, 0),  # Olive
        ]
        
    def initialize_camera(self):
        """
        Inisialisasi kamera Dino-Lite dengan berbagai percobaan
        """
        print("🔍 Mencari kamera Dino-Lite...")
        
        # Coba beberapa index kamera
        for i in range(4):  # Coba 0-3
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Gunakan DSHOW untuk Windows
            
            if cap.isOpened():
                # Coba baca frame untuk memastikan
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Kamera ditemukan di index {i}")
                    self.camera_index = i
                    self.cap = cap
                    
                    # Coba atur properti khusus Dino-Lite
                    self._setup_dinolite_properties()
                    return True
                else:
                    cap.release()
        
        print("❌ Tidak dapat menemukan kamera Dino-Lite")
        return False
    
    def _setup_dinolite_properties(self):
        """
        Setup properti khusus untuk Dino-Lite
        """
        try:
            # Set Auto Exposure (biasanya -1 untuk auto, 1 untuk manual)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Mode manual sebagian
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            
            # Set Brightness dan Contrast
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100.0)
            
            # Coba set FPS yang stabil
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Set resolusi maksimum
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            print("   Pengaturan Dino-Lite diterapkan")
            self.is_dinolite = True
            
        except Exception as e:
            print(f"   Tidak dapat mengatur properti Dino-Lite: {e}")
            self.is_dinolite = False
    
    def load_model(self):
        """Load model YOLOv8"""
        print("\n🤖 Loading YOLOv8 model...")
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded: {len(self.model.names)} classes")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_output_directories(self):
        """Buat direktori untuk output"""
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("measurements", exist_ok=True)
    
    def add_measurement_overlay(self, frame, detections):
        """
        Tambahkan overlay pengukuran untuk Dino-Lite
        """
        height, width = frame.shape[:2]
        
        # Grid overlay (opsional)
        if False:  # Ganti True untuk menampilkan grid
            grid_size = 50
            for x in range(0, width, grid_size):
                cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
            for y in range(0, height, grid_size):
                cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
        
        # Center crosshair
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        
        # Scale bar (asumsi kalibrasi: 100 pixel = 1mm)
        scale_length = 100  # pixel
        scale_start = (width - 150, height - 50)
        scale_end = (scale_start[0] + scale_length, scale_start[1])
        
        cv2.line(frame, scale_start, scale_end, (255, 255, 255), 3)
        cv2.putText(frame, "1 mm", (scale_start[0] - 10, scale_start[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Informasi deteksi dengan pengukuran
        if len(detections) > 0:
            for i, detection in enumerate(detections[:5]):  # Tampilkan 5 deteksi pertama
                cls_id, conf, box = detection
                class_name = self.model.names[int(cls_id)]
                
                # Hitung ukuran bounding box (dalam pixel)
                x1, y1, x2, y2 = box
                width_px = x2 - x1
                height_px = y2 - y1
                area_px = width_px * height_px
                
                # Konversi ke mm (asumsi: 100 pixel = 1mm)
                width_mm = width_px / 100.0
                height_mm = height_px / 100.0
                area_mm2 = area_px / 10000.0
                
                # Tampilkan informasi
                info_y = 100 + i * 80
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i % len(self.colors)], 2)
                cv2.putText(frame, f"Size: {width_mm:.2f} x {height_mm:.2f} mm", (10, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Area: {area_mm2:.2f} mm²", (10, info_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def apply_zoom_and_pan(self, frame):
        """Terapkan zoom dan pan pada frame"""
        if self.zoom_factor == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Hitung ukuran setelah zoom
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)
        
        # Hitung area untuk crop
        center_x = w // 2 + self.pan_x
        center_y = h // 2 + self.pan_y
        
        x1 = max(0, center_x - new_w // 2)
        x2 = min(w, center_x + new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        y2 = min(h, center_y + new_h // 2)
        
        # Crop frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize kembali ke ukuran asli
        if cropped.size > 0:
            return cv2.resize(cropped, (w, h))
        
        return frame
    
    def process_frame(self, frame):
        """Proses satu frame untuk deteksi"""
        # Terapkan zoom dan pan
        frame = self.apply_zoom_and_pan(frame)
        
        # Deteksi dengan YOLOv8
        results = self.model(frame, 
                           conf=self.confidence_threshold,
                           iou=self.iou_threshold,
                           verbose=False)
        
        # Annotasi frame dengan deteksi
        annotated_frame = results[0].plot()
        
        # Ambil informasi deteksi
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append((cls_id, conf, box))
                self.detection_count += 1
        
        # Tambahkan overlay pengukuran
        annotated_frame = self.add_measurement_overlay(annotated_frame, detections)
        
        # Tambahkan informasi status
        annotated_frame = self.add_status_overlay(annotated_frame)
        
        return annotated_frame, detections
    
    def add_status_overlay(self, frame):
        """Tambahkan overlay status"""
        # Hitung FPS
        current_time = time.time()
        if self.prev_time > 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Background untuk status bar
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
        
        # Status bar atas
        status_texts = [
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Zoom: {self.zoom_factor:.1f}x",
            f"Conf: {self.confidence_threshold:.2f}"
        ]
        
        x_pos = 10
        for text in status_texts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(frame, text, (x_pos, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x_pos += text_size[0] + 20
        
        # Status bar bawah (kontrol)
        control_texts = [
            "Q: Quit",
            "F: Fullscreen",
            "S: Screenshot",
            "R: Record",
            "+/-: Zoom",
            "C: Calibrate",
            "L: Lighting",
            "Z: Reset Zoom"
        ]
        
        x_pos = 10
        for text in control_texts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, text, (x_pos, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x_pos += text_size[0] + 15
        
        # Tampilkan jika sedang recording
        if self.recording:
            cv2.circle(frame, (w-20, 15), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w-50, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def start_recording(self):
        """Mulai recording video"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/dinolite_{timestamp}.avi"
            
            # Get frame size
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 
                                                   self.record_fps, (w, h))
                self.recording = True
                print(f"🎥 Started recording: {filename}")
    
    def stop_recording(self):
        """Stop recording video"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("⏹️ Recording stopped")
    
    def save_screenshot(self, frame):
        """Simpan screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/dinolite_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def adjust_lighting(self):
        """Menu adjustment pencahayaan"""
        print("\n💡 Lighting Adjustment:")
        print("   B: Brightness (+/-)")
        print("   C: Contrast (+/-)")
        print("   E: Exposure (+/-)")
        print("   X: Exit lighting menu")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('x'):
                break
            elif key == ord('b'):  # Brightness
                self.brightness = min(100, max(0, self.brightness + 10))
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100.0)
                print(f"   Brightness: {self.brightness}")
            elif key == ord('c'):  # Contrast
                self.contrast = min(100, max(0, self.contrast + 10))
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100.0)
                print(f"   Contrast: {self.contrast}")
            elif key == ord('e'):  # Exposure
                self.exposure = min(100, max(-10, self.exposure + 1))
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                print(f"   Exposure: {self.exposure}")
    
    def run(self):
        """Jalankan aplikasi utama"""
        print("=" * 60)
        print("DINO-LITE YOLOv8 DETECTION SYSTEM")
        print("=" * 60)
        
        # Inisialisasi
        if not self.initialize_camera():
            return
        
        if not self.load_model():
            return
        
        self.create_output_directories()
        
        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        print("\n🎮 CONTROLS:")
        print("   Q : Quit")
        print("   F : Toggle Fullscreen")
        print("   S : Screenshot")
        print("   R : Start/Stop Recording")
        print("   + : Zoom In")
        print("   - : Zoom Out")
        print("   Z : Reset Zoom/Pan")
        print("   Arrow Keys : Pan View")
        print("   L : Lighting Settings")
        print("   1/2 : Decrease/Increase Confidence")
        print("   C : Calibration Mode")
        print("=" * 60)
        
        print("\n🚀 Starting Dino-Lite detection...")
        
        try:
            while True:
                # Baca frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Proses frame
                processed_frame, detections = self.process_frame(frame)
                
                # Tampilkan
                cv2.imshow(self.window_name, processed_frame)
                
                # Simpan ke video jika recording
                if self.recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Update frame count
                self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q atau ESC
                    break
                elif key == ord('f'):  # Fullscreen
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(self.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_NORMAL)
                elif key == ord('s'):  # Screenshot
                    self.save_screenshot(processed_frame)
                elif key == ord('r'):  # Recording
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord('+'):  # Zoom in
                    self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
                elif key == ord('-'):  # Zoom out
                    self.zoom_factor = max(1.0, self.zoom_factor - 0.1)
                elif key == ord('z'):  # Reset zoom/pan
                    self.zoom_factor = 1.0
                    self.pan_x, self.pan_y = 0, 0
                elif key == ord('l'):  # Lighting
                    self.adjust_lighting()
                elif key == ord('1'):  # Decrease confidence
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                elif key == ord('2'):  # Increase confidence
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                elif key == ord('c'):  # Calibration
                    print("🔧 Entering calibration mode...")
                    self.calibrate_measurement()
                
                # Arrow keys for panning
                elif key == 82:  # Up arrow
                    self.pan_y -= 10
                elif key == 84:  # Down arrow
                    self.pan_y += 10
                elif key == 81:  # Left arrow
                    self.pan_x -= 10
                elif key == 83:  # Right arrow
                    self.pan_x += 10
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            self.cleanup()
    
    def calibrate_measurement(self):
        """Mode kalibrasi untuk pengukuran"""
        print("\n📏 Calibration Mode")
        print("   Place a known object (e.g., 1mm scale) in view")
        print("   Press 'C' to capture reference")
        print("   Press 'X' to exit calibration")
        
        ret, frame = self.cap.read()
        if ret:
            # Tampilkan frame untuk kalibrasi
            cv2.putText(frame, "CALIBRATION MODE", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.putText(frame, "Place reference object and press 'C'", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.window_name, frame)
    
    def cleanup(self):
        """Bersihkan resources"""
        if self.recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY:")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Total Detections: {self.detection_count}")
        print(f"   Average FPS: {self.fps:.1f}")
        print("=" * 60)
        print("✅ Program terminated successfully")

# Program utama
if __name__ == "__main__":
    # Path ke model Anda
    MODEL_PATH = r"D:\on\Project_Artificial_Intelegent\Computer_Vision\Otics_Plant1_Common_Rail\best.pt"
    
    # Jalankan aplikasi
    app = DinoLiteYOLODetector(MODEL_PATH)
    app.run()
