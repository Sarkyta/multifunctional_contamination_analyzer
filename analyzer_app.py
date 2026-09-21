import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import os
import datetime
import logging
import glob
import threading
import queue
from typing import Optional, Tuple, Dict, Any, List
import math
import traceback

# Настройка логов
try:
    logging.basicConfig(
        filename='Result.log',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        encoding='utf-8'
    )
except Exception:
    logging.basicConfig(
        filename='Result.log',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

class ModernTheme:
    """Современная цветовая схема приложения"""
    COLORS = {
        'primary': '#2c3e50',
        'secondary': '#34495e',
        'accent': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'light': '#ecf0f1',
        'dark': '#2c3e50',
        'text': '#2c3e50',
        'text_light': '#7f8c8d',
        'background': '#f8f9fa',
        'panel': '#ffffff',
        'border': '#d1d8e0'
    }
    FONTS = {
        'title': ('Arial', 16, 'bold'),
        'heading': ('Arial', 12, 'bold'),
        'normal': ('Arial', 10),
        'small': ('Arial', 9),
        'button': ('Arial', 10, 'bold')
    }

class ThreadSafeLogger:
    """Потокобезопасный логгер"""
    def __init__(self):
        self.log_queue = queue.Queue()
        self.root = None

    def set_root(self, root):
        self.root = root
        self.process_queue()

    def log(self, level, message):
        self.log_queue.put((level, message))
        if self.root:
            self.root.after(0, self.process_queue)

    def process_queue(self):
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                if level == 'INFO':
                    logging.info(message)
                elif level == 'ERROR':
                    logging.error(message)
                elif level == 'WARNING':
                    logging.warning(message)
        except queue.Empty:
            pass
        if self.root:
            self.root.after(100, self.process_queue)

# Глобальный потокобезопасный логгер
thread_safe_logger = ThreadSafeLogger()

class ImageCache:
    """Кэш для управления изображениями и предотвращения утечек памяти"""
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()

class AdvancedImageProcessor:
    @staticmethod
    def enhanced_preprocessing(img: np.ndarray) -> np.ndarray:
        """Расширенная предобработка изображения"""
        try:
            if img is None or img.size == 0:
                return img

            processed = img.copy()
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
            return processed
        except Exception as e:
            thread_safe_logger.log('WARNING', f"Ошибка расширенной предобработки: {e}")
            return img

    @staticmethod
    def postprocess_mask(mask: np.ndarray, min_area: int = 100, kernel_size: int = 5) -> np.ndarray:
        """Постобработка маски для удаления шума"""
        try:
            if mask is None or mask.size == 0:
                return mask

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    cv2.fillPoly(mask, [contour], 0)
            return mask
        except Exception as e:
            thread_safe_logger.log('WARNING', f"Ошибка постобработки маски: {e}")
            return mask

class AreaCalculator:
    @staticmethod
    def calculate_area_stats(mask: np.ndarray, image_shape: tuple,
                           scale_mm_per_pixel: Optional[float] = None,
                           width_mm: float = 100.0, height_mm: float = 100.0) -> Dict[str, Any]:
        """Расчет статистики площади с улучшенной точностью"""
        try:
            if mask is None or mask.size == 0:
                return {}

            h, w = image_shape
            total_pixels = h * w

            if total_pixels == 0:
                return {}

            contaminated_px = cv2.countNonZero(mask)
            percent = (contaminated_px / total_pixels) * 100 if total_pixels > 0 else 0

            if scale_mm_per_pixel is not None:
                pixel_area_mm2 = scale_mm_per_pixel ** 2
                contaminated_mm2 = contaminated_px * pixel_area_mm2
                total_area_mm2 = total_pixels * pixel_area_mm2
            else:
                total_area_mm2 = width_mm * height_mm
                contaminated_mm2 = (contaminated_px / total_pixels) * total_area_mm2

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)

            largest_area_mm2 = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                if scale_mm_per_pixel is not None:
                    largest_area_mm2 = largest_area * (scale_mm_per_pixel ** 2)
                else:
                    largest_area_mm2 = (largest_area / total_pixels) * total_area_mm2

            return {
                'contaminated_px': contaminated_px,
                'total_pixels': total_pixels,
                'percent': percent,
                'contaminated_mm2': contaminated_mm2,
                'total_area_mm2': total_area_mm2,
                'num_contours': num_contours,
                'largest_contour_area_mm2': largest_area_mm2
            }
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка расчета площади: {e}")
            return {}

class ImageDisplayManager:
    """Менеджер для отображения изображений с поддержкой масштабирования"""
    def __init__(self, original_label, mask_label):
        self.original_label = original_label
        self.mask_label = mask_label
        self.original_image = None
        self.mask_image = None
        self.display_scale = 1.0
        self.original_offset_x = 0
        self.original_offset_y = 0
        self.mask_offset_x = 0
        self.mask_offset_y = 0

    def display_to_original_coords(self, x, y):
        """Конвертация координат отображения в оригинальные координаты"""
        if self.original_image is None:
            return x, y

        orig_x = int((x - self.original_offset_x) / self.display_scale)
        orig_y = int((y - self.original_offset_y) / self.display_scale)
        return orig_x, orig_y

    def set_original_image(self, image):
        """Установка оригинального изображения"""
        self.original_image = image
        self.update_display()

    def set_mask_image(self, image):
        """Установка маски"""
        self.mask_image = image
        self.update_display()

    def update_display(self):
        """Обновление отображения"""
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_label, "original")
        if self.mask_image is not None:
            self.display_image(self.mask_image, self.mask_label, "mask")

    def display_image(self, img: np.ndarray, label: tk.Label, cache_key: str):
        """Отображение изображения с поддержкой масштабирования"""
        try:
            if img is None or img.size == 0:
                label.configure(image='', text="Ошибка отображения")
                return

            if len(img.shape) == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            h, w = rgb.shape[:2]

            # Получаем размеры метки для правильного масштабирования
            label_width = label.winfo_width()
            label_height = label.winfo_height()

            if label_width > 1 and label_height > 1:
                scale_x = label_width / w
                scale_y = label_height / h
                self.display_scale = min(scale_x, scale_y, 1.0)
            else:
                # Fallback если размеры еще не известны
                max_h = 500
                self.display_scale = min(1.0, max_h / h)

            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Центрирование изображения
            self.original_offset_x = (label_width - new_w) // 2 if label_width > new_w else 0
            self.original_offset_y = (label_height - new_h) // 2 if label_height > new_h else 0

            pil_img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(pil_img)

            label.configure(image=imgtk, text="")
            label.image = imgtk

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка отображения изображения: {e}")
            label.configure(image='', text="Ошибка отображения изображения")

class CompactHSVRangeVisualizer:
    """Компактный визуализатор диапазона HSV"""
    def __init__(self, parent, width=600):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, pady=5)

        self.canvas = tk.Canvas(self.frame, height=120, bg='white',
                               highlightthickness=1, highlightbackground="#cccccc",
                               width=width)
        self.canvas.pack(fill=tk.X, padx=5, pady=5)

        self.info_label = ttk.Label(self.frame, text="Диапазон HSV: H(0-179) S(0-255) V(0-255)")
        self.info_label.pack(fill=tk.X)

        self.update_range(35, 85, 50, 255, 50, 255)

    def update_range(self, h_min, h_max, s_min, s_max, v_min, v_max):
        """Обновление визуализации диапазона"""
        try:
            self.canvas.delete("all")
            canvas_width = self.canvas.winfo_width() - 10
            if canvas_width < 100:
                canvas_width = 550

            bar_height = 25
            margin = 5
            bar_width = canvas_width

            # Hue
            y_hue = margin
            self.canvas.create_text(margin, y_hue - 5, text="H", anchor=tk.W, font=('Arial', 8, 'bold'))
            for i in range(bar_width):
                hue = int(i * 179 / bar_width)
                x = margin + i
                color = self.hsv_to_rgb(hue, 255, 255)
                self.canvas.create_line(x, y_hue, x, y_hue + bar_height, fill=color)

            h_start = margin + (h_min / 179) * bar_width
            h_end = margin + (h_max / 179) * bar_width
            self.canvas.create_rectangle(h_start, y_hue, h_end, y_hue + bar_height,
                                       outline='red', width=2)

            # Saturation
            y_sat = y_hue + bar_height + margin + 5
            self.canvas.create_text(margin, y_sat - 5, text="S", anchor=tk.W, font=('Arial', 8, 'bold'))
            for i in range(bar_width):
                sat = int(i * 255 / bar_width)
                x = margin + i
                color = self.hsv_to_rgb((h_min + h_max) // 2, sat, 255)
                self.canvas.create_line(x, y_sat, x, y_sat + bar_height, fill=color)

            s_start = margin + (s_min / 255) * bar_width
            s_end = margin + (s_max / 255) * bar_width
            self.canvas.create_rectangle(s_start, y_sat, s_end, y_sat + bar_height,
                                       outline='red', width=2)

            # Value
            y_val = y_sat + bar_height + margin + 5
            self.canvas.create_text(margin, y_val - 5, text="V", anchor=tk.W, font=('Arial', 8, 'bold'))
            for i in range(bar_width):
                val = int(i * 255 / bar_width)
                x = margin + i
                color = self.hsv_to_rgb((h_min + h_max) // 2, 255, val)
                self.canvas.create_line(x, y_val, x, y_val + bar_height, fill=color)

            v_start = margin + (v_min / 255) * bar_width
            v_end = margin + (v_max / 255) * bar_width
            self.canvas.create_rectangle(v_start, y_val, v_end, y_val + bar_height,
                                       outline='red', width=2)

            self.info_label.config(text=f"Диапазон HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка обновления визуализатора HSV: {e}")

    def hsv_to_rgb(self, h, s, v):
        """Конвертация HSV в RGB цвет для Tkinter"""
        try:
            hsv_color = np.uint8([[[h, s, v]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
            r, g, b = rgb_color[0][0]
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception:
            return '#000000'

class BatchAnalysisTool:
    """Инструмент для пакетного анализа изображений"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.window = None
        self.progress_queue = queue.Queue()
        self.is_processing = False
        self.current_file_index = 0
        self.total_files = 0
        self.setup_ui()

    def setup_ui(self):
        """Настройка интерфейса пакетного анализа"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("📊 Пакетный анализ изображений")
        self.window.geometry("800x600")
        self.window.transient(self.parent)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = ttk.Label(main_frame,
                               text="📊 Пакетный анализ изображений",
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(pady=(0, 15))

        # Секция выбора файлов
        file_selection_frame = ttk.LabelFrame(main_frame, text="📁 Выбор изображений", padding=15)
        file_selection_frame.pack(fill=tk.X, pady=(0, 15))

        # Кнопки выбора файлов
        button_frame = ttk.Frame(file_selection_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="➕ Добавить файлы",
                  command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📁 Добавить папку",
                  command=self.add_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗑️ Очистить список",
                  command=self.clear_files).pack(side=tk.LEFT, padx=5)

        # Список файлов
        list_frame = ttk.Frame(file_selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Заголовки списка
        header_frame = ttk.Frame(list_frame)
        header_frame.pack(fill=tk.X)

        ttk.Label(header_frame, text="Файл", width=40, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Размер", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Статус", width=15, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)

        # Прокручиваемый список файлов
        self.file_list_frame = ttk.Frame(list_frame)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(self.file_list_frame, height=150)
        scrollbar = ttk.Scrollbar(self.file_list_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Секция параметров анализа
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Параметры анализа", padding=15)
        settings_frame.pack(fill=tk.X, pady=(0, 15))

        # Использовать текущие настройки HSV
        self.use_current_hsv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, 
                       text="Использовать текущие настройки HSV для всех изображений",
                       variable=self.use_current_hsv_var).pack(anchor=tk.W, pady=5)

        # Авто-калибровка для каждого изображения
        self.auto_calibrate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame,
                       text="Автоматическая калибровка цвета для каждого изображения",
                       variable=self.auto_calibrate_var).pack(anchor=tk.W, pady=5)

        # Сохранять обработанные изображения
        self.save_processed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame,
                       text="Сохранять обработанные изображения с результатами",
                       variable=self.save_processed_var).pack(anchor=tk.W, pady=5)

        # Секция прогресса
        progress_frame = ttk.LabelFrame(main_frame, text="📈 Прогресс анализа", padding=15)
        progress_frame.pack(fill=tk.X, pady=(0, 15))

        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=700)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Статус
        self.status_label = ttk.Label(progress_frame, 
                                     text="Готов к анализу",
                                     font=('Arial', 10))
        self.status_label.pack(fill=tk.X, pady=5)

        # Детали прогресса
        self.details_label = ttk.Label(progress_frame,
                                      text="Обработано: 0 из 0 файлов",
                                      foreground='#7f8c8d')
        self.details_label.pack(fill=tk.X)

        # Секция результатов
        results_frame = ttk.LabelFrame(main_frame, text="📋 Результаты", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Текстовое поле для логов
        self.log_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Кнопки управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(control_frame, 
                                      text="🚀 Начать пакетный анализ",
                                      command=self.start_batch_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, 
                  text="💾 Экспорт результатов",
                  command=self.export_results).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, 
                  text="❌ Закрыть",
                  command=self.on_closing).pack(side=tk.RIGHT, padx=5)

        # Инициализация переменных
        self.file_list = []
        self.results = []
        self.processing_thread = None

        # Запуск обработки очереди прогресса
        self.process_queue()

    def add_files(self):
        """Добавление отдельных файлов"""
        filepaths = filedialog.askopenfilenames(
            title="Выберите изображения для анализа",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
        )

        if filepaths:
            for filepath in filepaths:
                if filepath not in [f['path'] for f in self.file_list]:
                    self.add_file_to_list(filepath)
            self.update_file_count()

    def add_folder(self):
        """Добавление всех изображений из папки"""
        folder_path = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder_path:
            supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
            for format in supported_formats:
                files = glob.glob(os.path.join(folder_path, format))
                files.extend(glob.glob(os.path.join(folder_path, format.upper())))
                for filepath in files:
                    if filepath not in [f['path'] for f in self.file_list]:
                        self.add_file_to_list(filepath)
            self.update_file_count()

    def add_file_to_list(self, filepath):
        """Добавление файла в список"""
        try:
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)

            file_info = {
                'path': filepath,
                'name': os.path.basename(filepath),
                'size_mb': file_size_mb,
                'status': 'В ожидании'
            }
            self.file_list.append(file_info)

            # Добавление в UI
            self.add_file_to_ui(file_info)

        except Exception as e:
            self.log_message(f"Ошибка добавления файла {filepath}: {str(e)}")

    def add_file_to_ui(self, file_info):
        """Добавление файла в пользовательский интерфейс"""
        file_frame = ttk.Frame(self.scrollable_frame)
        file_frame.pack(fill=tk.X, pady=2)

        ttk.Label(file_frame, text=file_info['name'], width=40).pack(side=tk.LEFT)
        ttk.Label(file_frame, text=f"{file_info['size_mb']:.2f} МБ", width=10).pack(side=tk.LEFT)

        status_label = ttk.Label(file_frame, text=file_info['status'], width=15)
        status_label.pack(side=tk.LEFT)

        # Сохраняем ссылку на элемент для обновления статуса
        file_info['ui_status'] = status_label

    def clear_files(self):
        """Очистка списка файлов"""
        self.file_list.clear()
        # Очистка UI
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.update_file_count()

    def update_file_count(self):
        """Обновление счетчика файлов"""
        self.details_label.config(text=f"Всего файлов: {len(self.file_list)}")

    def start_batch_analysis(self):
        """Запуск пакетного анализа"""
        if not self.file_list:
            messagebox.showwarning("Ошибка", "Добавьте файлы для анализа")
            return

        if self.is_processing:
            messagebox.showwarning("Ошибка", "Анализ уже выполняется")
            return

        self.is_processing = True
        self.start_button.config(state='disabled')
        self.results = []
        self.current_file_index = 0
        self.total_files = len(self.file_list)

        # Запуск в отдельном потоке
        self.processing_thread = threading.Thread(target=self.process_batch, daemon=True)
        self.processing_thread.start()

    def process_batch(self):
        """Обработка пакета изображений"""
        try:
            self.log_message("🚀 Начало пакетного анализа...")
            self.update_progress(0, "Подготовка к анализу")

            for i, file_info in enumerate(self.file_list):
                if not self.is_processing:
                    break

                self.current_file_index = i
                file_info['status'] = 'В процессе'
                self.update_file_status(file_info)

                try:
                    result = self.process_single_file(file_info)
                    self.results.append(result)

                    file_info['status'] = '✅ Завершено'
                    self.update_file_status(file_info)

                    self.log_message(f"✅ Обработан: {file_info['name']} - {result['percent']:.2f}% загрязнения")

                except Exception as e:
                    file_info['status'] = '❌ Ошибка'
                    self.update_file_status(file_info)
                    self.log_message(f"❌ Ошибка обработки {file_info['name']}: {str(e)}")

                # Обновление прогресса
                progress = (i + 1) / len(self.file_list) * 100
                self.update_progress(progress, f"Обработка {i + 1}/{len(self.file_list)}")

            if self.is_processing:
                self.update_progress(100, "Анализ завершен")
                self.log_message("🎉 Пакетный анализ успешно завершен!")
                self.complete_analysis()

        except Exception as e:
            self.log_message(f"❌ Критическая ошибка пакетного анализа: {str(e)}")
            self.update_progress(0, "Анализ прерван из-за ошибки")

        finally:
            self.is_processing = False
            self.start_button.config(state='normal')

    def process_single_file(self, file_info):
        """Обработка одного файла"""
        # Загрузка изображения
        image = cv2.imread(file_info['path'])
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        # Сохраняем текущие настройки
        original_hsv_lower = self.app.hsv_lower.copy()
        original_hsv_upper = self.app.hsv_upper.copy()

        try:
            # Авто-калибровка если включена
            if self.auto_calibrate_var.get():
                self.auto_calibrate_image(image)

            # Применяем улучшенную предобработку
            processed_img = self.app.advanced_processor.enhanced_preprocessing(image)

            # Создание маски
            hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.app.hsv_lower, self.app.hsv_upper)

            if mask is None or cv2.countNonZero(mask) == 0:
                raise ValueError("Не обнаружено областей загрязнения")

            # Постобработка маски
            mask = self.app.advanced_processor.postprocess_mask(
                mask,
                min_area=self.app.min_area_var.get(),
                kernel_size=self.app.kernel_size_var.get()
            )

            # Расчет статистики
            stats = self.app.area_calculator.calculate_area_stats(
                mask,
                image.shape[:2],
                self.app.scale_mm_per_pixel,
                self.app.width_var.get(),
                self.app.height_var.get()
            )

            if not stats:
                raise ValueError("Не удалось рассчитать статистику")

            # Сохранение обработанного изображения если включено
            output_image_path = None
            if self.save_processed_var.get():
                output_image_path = self.save_processed_image(image, mask, file_info)

            # Формирование результата
            result = {
                'filename': file_info['name'],
                'filepath': file_info['path'],
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'contaminated_area_mm2': stats['contaminated_mm2'],
                'total_area_mm2': stats['total_area_mm2'],
                'contamination_percent': stats['percent'],
                'contamination_pixels': stats['contaminated_px'],
                'total_pixels': stats['total_pixels'],
                'num_contours': stats['num_contours'],
                'largest_contour_mm2': stats['largest_contour_area_mm2'],
                'output_image': output_image_path,
                'hsv_range': f"H({self.app.hsv_lower[0]}-{self.app.hsv_upper[0]}) "
                           f"S({self.app.hsv_lower[1]}-{self.app.hsv_upper[1]}) "
                           f"V({self.app.hsv_lower[2]}-{self.app.hsv_upper[2]})"
            }

            return result

        finally:
            # Восстанавливаем оригинальные настройки HSV если не используем текущие
            if not self.use_current_hsv_var.get():
                self.app.hsv_lower = original_hsv_lower
                self.app.hsv_upper = original_hsv_upper

    def auto_calibrate_image(self, image):
        """Автоматическая калибровка для отдельного изображения"""
        try:
            # Упрощенная автоматическая калибровка
            # В реальной реализации здесь может быть более сложный алгоритм
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Анализ гистограммы для определения доминирующих цветов
            h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

            # Находим пики в гистограммах
            h_peak = np.argmax(h_hist)
            s_peak = np.argmax(s_hist)
            v_peak = np.argmax(v_hist)

            # Устанавливаем диапазон вокруг пиков
            h_range = 30
            s_range = 60
            v_range = 60

            self.app.hsv_lower = np.array([
                max(0, h_peak - h_range),
                max(50, s_peak - s_range),
                max(50, v_peak - v_range)
            ])
            self.app.hsv_upper = np.array([
                min(179, h_peak + h_range),
                min(255, s_peak + s_range),
                min(255, v_peak + v_range)
            ])

        except Exception as e:
            self.log_message(f"⚠️ Ошибка авто-калибровки: {str(e)}")
            # Используем настройки по умолчанию в случае ошибки
            self.app.hsv_lower = np.array([35, 50, 50])
            self.app.hsv_upper = np.array([85, 255, 255])

    def save_processed_image(self, image, mask, file_info):
        """Сохранение обработанного изображения"""
        try:
            # Создаем папку для результатов если не существует
            results_dir = os.path.join(os.path.dirname(file_info['path']), "batch_analysis_results")
            os.makedirs(results_dir, exist_ok=True)

            # Создаем визуализацию с маской
            result_img = image.copy()
            color_map = {
                'red': [0, 0, 255],
                'green': [0, 255, 0],
                'blue': [255, 0, 0],
                'yellow': [0, 255, 255],
                'cyan': [255, 255, 0],
                'magenta': [255, 0, 255]
            }
            highlight_color = color_map.get(self.app.contour_color_var.get(), [0, 0, 255])
            result_img[mask > 0] = highlight_color

            # Сохраняем изображение
            filename, ext = os.path.splitext(file_info['name'])
            output_path = os.path.join(results_dir, f"{filename}_processed{ext}")
            cv2.imwrite(output_path, result_img)

            return output_path

        except Exception as e:
            self.log_message(f"⚠️ Не удалось сохранить обработанное изображение: {str(e)}")
            return None

    def update_progress(self, value, status):
        """Обновление прогресса через очередь"""
        self.progress_queue.put(('progress', value, status))

    def update_file_status(self, file_info):
        """Обновление статуса файла через очередь"""
        self.progress_queue.put(('file_status', file_info))

    def log_message(self, message):
        """Добавление сообщения в лог через очередь"""
        self.progress_queue.put(('log', message))

    def process_queue(self):
        """Обработка очереди обновлений UI"""
        try:
            while True:
                item = self.progress_queue.get_nowait()
                item_type = item[0]

                if item_type == 'progress':
                    value, status = item[1], item[2]
                    self.progress_var.set(value)
                    self.status_label.config(text=status)
                    self.details_label.config(text=f"Обработано: {self.current_file_index + 1} из {self.total_files} файлов")

                elif item_type == 'file_status':
                    file_info = item[1]
                    if 'ui_status' in file_info:
                        file_info['ui_status'].config(text=file_info['status'])

                elif item_type == 'log':
                    message = item[1]
                    self.log_text.insert(tk.END, f"{message}\n")
                    self.log_text.see(tk.END)

        except queue.Empty:
            pass

        # Продолжаем обработку очереди
        if self.window and self.window.winfo_exists():
            self.window.after(100, self.process_queue)

    def complete_analysis(self):
        """Завершение анализа"""
        self.log_message(f"📊 Анализ завершен. Обработано {len(self.results)} из {len(self.file_list)} файлов")

        # Сводная статистика
        if self.results:
            total_contamination = sum(r['contamination_percent'] for r in self.results)
            avg_contamination = total_contamination / len(self.results)
            self.log_message(f"📈 Средний процент загрязнения: {avg_contamination:.2f}%")

    def export_results(self):
        """Экспорт результатов в файл"""
        if not self.results:
            messagebox.showwarning("Ошибка", "Нет данных для экспорта")
            return

        try:
            # Пробуем импортировать pandas
            try:
                import pandas as pd
                pandas_available = True
            except ImportError:
                pandas_available = False
                self.log_message("⚠️ Библиотека pandas не установлена. Экспорт в CSV будет доступен без pandas.")

            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )

            if filename:
                if pandas_available and filename.endswith('.csv'):
                    # Используем pandas для экспорта
                    df_data = []
                    for result in self.results:
                        df_data.append({
                            'Файл': result['filename'],
                            'Время анализа': result['timestamp'],
                            'Площадь загрязнения (мм²)': result['contaminated_area_mm2'],
                            'Общая площадь (мм²)': result['total_area_mm2'],
                            'Процент загрязнения (%)': result['contamination_percent'],
                            'Пикселей загрязнения': result['contamination_pixels'],
                            'Всего пикселей': result['total_pixels'],
                            'Количество областей': result['num_contours'],
                            'Площадь крупнейшей области (мм²)': result['largest_contour_mm2'],
                            'Диапазон HSV': result['hsv_range'],
                            'Обработанное изображение': result['output_image'] or 'Не сохранено'
                        })

                    df = pd.DataFrame(df_data)
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    
                else:
                    # Простой текстовый экспорт
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("Результаты пакетного анализа загрязнений\n")
                        f.write("=" * 50 + "\n\n")
                        
                        for result in self.results:
                            f.write(f"Файл: {result['filename']}\n")
                            f.write(f"Время анализа: {result['timestamp']}\n")
                            f.write(f"Площадь загрязнения: {result['contaminated_area_mm2']:.2f} мм²\n")
                            f.write(f"Общая площадь: {result['total_area_mm2']:.2f} мм²\n")
                            f.write(f"Процент загрязнения: {result['contamination_percent']:.2f}%\n")
                            f.write(f"Пикселей загрязнения: {result['contamination_pixels']}\n")
                            f.write(f"Всего пикселей: {result['total_pixels']}\n")
                            f.write(f"Количество областей: {result['num_contours']}\n")
                            f.write(f"Площадь крупнейшей области: {result['largest_contour_mm2']:.2f} мм²\n")
                            f.write(f"Диапазон HSV: {result['hsv_range']}\n")
                            f.write(f"Обработанное изображение: {result['output_image'] or 'Не сохранено'}\n")
                            f.write("-" * 30 + "\n\n")

                self.log_message(f"💾 Результаты экспортированы в: {filename}")
                messagebox.showinfo("Успех", "Результаты успешно экспортированы")

        except Exception as e:
            error_msg = f"Ошибка экспорта: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("Ошибка", error_msg)

    def on_closing(self):
        """Обработка закрытия окна"""
        if self.is_processing:
            if messagebox.askokcancel("Выход", "Анализ все еще выполняется. Прервать?"):
                self.is_processing = False
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=2.0)
                self.window.destroy()
        else:
            self.window.destroy()

class RulerTool:
    def __init__(self, parent, image, callback):
        self.parent = parent
        self.image = image.copy()
        self.callback = callback
        self.start_point = None
        self.end_point = None
        self.window = tk.Toplevel(parent)
        self.window.title(" 📏 Установка масштаба")
        self.window.geometry("800x600")
        self.window.transient(parent)
        self.window.grab_set()
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Нарисуйте отрезок и укажите его длину в мм").pack(pady=5)

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Длина (мм):").pack(side=tk.LEFT)
        self.length_var = tk.DoubleVar(value=10.0)
        ttk.Entry(input_frame, textvariable=self.length_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(input_frame, text=" ✅ Применить", 
                  command=self.apply).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(main_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.display_image()

    def display_image(self):
        h, w = self.image.shape[:2]
        max_h = 500
        scale = min(1.0, max_h / h)
        self.display_scale = scale

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(self.image, (new_w, new_h))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(self.pil_img)

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_click(self, event):
        if self.start_point is None:
            self.start_point = (event.x, event.y)
        else:
            self.end_point = (event.x, event.y)
            self.canvas.delete("ruler")
            self.canvas.create_line(self.start_point, self.end_point, 
                                  fill='yellow', width=2, tags="ruler")
            self.start_point = None

    def apply(self):
        if self.end_point is None:
            messagebox.showwarning("Ошибка", "Сначала нарисуйте отрезок")
            return

        x1, y1 = self.start_point if self.start_point else (0, 0)
        x2, y2 = self.end_point
        pixel_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        real_length = self.length_var.get()
        if real_length <= 0:
            messagebox.showerror("Ошибка", "Длина должна быть > 0")
            return

        scale_mm_per_pixel = real_length / pixel_length
        self.callback(scale_mm_per_pixel)
        self.window.destroy()

class ColorCalibrationTool:
    """Улучшенный инструмент для калибровки цвета по областям изображения"""
    def __init__(self, parent, image, callback):
        self.parent = parent
        self.image = image.copy()
        self.callback = callback
        self.clean_roi = None
        self.dirty_roi = None
        self.mode = 'clean'
        self.rect_start = None
        self.window = tk.Toplevel(parent)
        self.window.title(" 🎨 Улучшенная калибровка цвета")
        self.window.geometry("900x700")
        self.window.transient(parent)
        self.window.grab_set()
        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        main_frame = ttk.Frame(self.window, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок и инструкция
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame, 
                               text="🎨 Калибровка цвета по областям", 
                               font=('Arial', 14, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(anchor=tk.W)

        instruction_text = (
            "1. Выберите ЧИСТУЮ область (без загрязнений)\n"
            "2. Выберите ЗАГРЯЗНЁННУЮ область\n"
            "3. Система автоматически определит оптимальный диапазон HSV"
        )
        instruction_label = ttk.Label(header_frame, 
                                     text=instruction_text,
                                     justify=tk.LEFT,
                                     foreground='#7f8c8d')
        instruction_label.pack(anchor=tk.W, pady=5)

        # Панель управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # Информация о текущем режиме
        self.mode_frame = ttk.Frame(control_frame)
        self.mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(self.mode_frame, 
                                     text="🔵 Выберите ЧИСТУЮ область (перетащите мышь)", 
                                     font=('Arial', 10, 'bold'),
                                     foreground='green')
        self.status_label.pack(anchor=tk.W)

        self.progress_label = ttk.Label(self.mode_frame,
                                       text="",
                                       foreground='#7f8c8d')
        self.progress_label.pack(anchor=tk.W)

        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)

        ttk.Button(button_frame, text=" 🔄 Начать заново", 
                  command=self.reset_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=" ✅ Применить калибровку", 
                  command=self.apply_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=" ❌ Отмена", 
                  command=self.window.destroy).pack(side=tk.LEFT, padx=5)

        # Область изображения
        image_frame = ttk.LabelFrame(main_frame, text="Изображение для калибровки", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Добавляем скроллбары для изображения
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

        self.canvas = tk.Canvas(canvas_frame, bg='black',
                               yscrollcommand=v_scrollbar.set,
                               xscrollcommand=h_scrollbar.set)

        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Привязка событий
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)

        # Информация о выбранных областях
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)

        self.info_label = ttk.Label(info_frame, 
                                   text="Области не выбраны",
                                   justify=tk.LEFT,
                                   foreground='#7f8c8d')
        self.info_label.pack(anchor=tk.W)

        self.display_image()

    def display_image(self):
        """Отображение изображения с поддержкой скроллинга"""
        try:
            if self.image is None:
                return

            h, w = self.image.shape[:2]

            # Масштабирование для отображения
            max_display_size = 800
            scale = min(1.0, max_display_size / max(h, w))
            self.display_scale = scale

            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(self.image, (new_w, new_h))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            self.pil_img = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(self.pil_img)

            # Установка размеров canvas
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Сохраняем размеры для преобразования координат
            self.canvas_width = new_w
            self.canvas_height = new_h

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка отображения изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось отобразить изображение: {str(e)}")

    def on_press(self, event):
        """Обработка нажатия кнопки мыши"""
        self.rect_start = (event.x, event.y)

    def on_drag(self, event):
        """Обработка перемещения мыши с зажатой кнопкой"""
        if self.rect_start is None:
            return

        self.canvas.delete("selection")
        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y

        # Определяем цвет рамки в зависимости от режима
        outline_color = 'lime' if self.mode == 'clean' else 'red'

        self.canvas.create_rectangle(x0, y0, x1, y1, 
                                   outline=outline_color, 
                                   width=3, 
                                   tags="selection")

    def on_release(self, event):
        """Обработка отпускания кнопки мыши"""
        if self.rect_start is None:
            return

        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y

        # Убеждаемся, что координаты упорядочены
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        # Проверяем минимальный размер области
        min_size = 10
        if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
            messagebox.showwarning("Предупреждение", 
                                 f"Выделенная область слишком мала. Минимальный размер: {min_size}px")
            self.canvas.delete("selection")
            self.rect_start = None
            return

        # Преобразование в оригинальные координаты
        orig_x1 = int(x_min / self.display_scale)
        orig_y1 = int(y_min / self.display_scale)
        orig_x2 = int(x_max / self.display_scale)
        orig_y2 = int(y_max / self.display_scale)

        if self.mode == 'clean':
            self.clean_roi = (orig_x1, orig_y1, orig_x2, orig_y2)
            self.mode = 'dirty'
            self.status_label.config(text="🔴 Теперь выберите ЗАГРЯЗНЁННУЮ область", 
                                   foreground='red')
            self.progress_label.config(text="✅ Чистая область выбрана")

        else:
            self.dirty_roi = (orig_x1, orig_y1, orig_x2, orig_y2)
            self.status_label.config(text="🟢 Калибровка готова. Нажмите «Применить калибровку».", 
                                   foreground='blue')
            self.progress_label.config(text="✅ Загрязненная область выбрана")

        self.update_info_display()
        self.rect_start = None

    def on_motion(self, event):
        """Обработка движения мыши для отображения координат"""
        if self.rect_start:
            x0, y0 = self.rect_start
            x1, y1 = event.x, event.y
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            orig_width = int(width / self.display_scale)
            orig_height = int(height / self.display_scale)

            self.progress_label.config(
                text=f"Размер области: {width}x{height} px (оригинал: {orig_width}x{orig_height} px)"
            )

    def update_info_display(self):
        """Обновление информации о выбранных областях"""
        info_text = ""

        if self.clean_roi:
            x1, y1, x2, y2 = self.clean_roi
            width = x2 - x1
            height = y2 - y1
            info_text += f"✅ Чистая область: ({x1}, {y1}) - ({x2}, {y2}) [{width}x{height}]\n"

        if self.dirty_roi:
            x1, y1, x2, y2 = self.dirty_roi
            width = x2 - x1
            height = y2 - y1
            info_text += f"✅ Загрязненная область: ({x1}, {y1}) - ({x2}, {y2}) [{width}x{height}]"

        if info_text:
            self.info_label.config(text=info_text, foreground='#2c3e50')
        else:
            self.info_label.config(text="Области не выбраны", foreground='#7f8c8d')

    def reset_selection(self):
        """Сброс выбранных областей"""
        self.clean_roi = None
        self.dirty_roi = None
        self.mode = 'clean'
        self.rect_start = None

        self.canvas.delete("selection")
        self.status_label.config(text="🔵 Выберите ЧИСТУЮ область (перетащите мышь)", 
                               foreground='green')
        self.progress_label.config(text="")
        self.update_info_display()

    def apply_calibration(self):
        """Применение калибровки с улучшенным алгоритмом"""
        if self.clean_roi is None or self.dirty_roi is None:
            messagebox.showwarning("Ошибка", "Необходимо выделить обе области: чистую и загрязненную")
            return

        try:
            # Извлечение областей
            clean_img = self.image[self.clean_roi[1]:self.clean_roi[3], 
                                  self.clean_roi[0]:self.clean_roi[2]]
            dirty_img = self.image[self.dirty_roi[1]:self.dirty_roi[3], 
                                  self.dirty_roi[0]:self.dirty_roi[2]]

            if clean_img.size == 0 or dirty_img.size == 0:
                messagebox.showerror("Ошибка", "Некорректные области - попробуйте выбрать другие области")
                return

            # Проверка минимального размера областей
            min_pixels = 100
            if clean_img.shape[0] * clean_img.shape[1] < min_pixels:
                messagebox.showwarning("Предупреждение", 
                                     "Чистая область слишком мала для точной калибровки")

            if dirty_img.shape[0] * dirty_img.shape[1] < min_pixels:
                messagebox.showwarning("Предупреждение", 
                                     "Загрязненная область слишком мала для точной калибровки")

            # Конвертация в HSV
            hsv_clean = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
            hsv_dirty = cv2.cvtColor(dirty_img, cv2.COLOR_BGR2HSV)

            # УЛУЧШЕННЫЙ АЛГОРИТМ ОПРЕДЕЛЕНИЯ ЦВЕТА

            # 1. Анализ чистой области для определения фона
            clean_h = hsv_clean[:, :, 0].flatten()
            clean_s = hsv_clean[:, :, 1].flatten()
            clean_v = hsv_clean[:, :, 2].flatten()

            # 2. Анализ загрязненной области с использованием устойчивых статистик
            dirty_h = hsv_dirty[:, :, 0].flatten()
            dirty_s = hsv_dirty[:, :, 1].flatten()
            dirty_v = hsv_dirty[:, :, 2].flatten()

            # Используем percentiles для лучшего определения диапазона
            h_low, h_high = np.percentile(dirty_h, [10, 90])
            s_low, s_high = np.percentile(dirty_s, [10, 90])
            v_low, v_high = np.percentile(dirty_v, [10, 90])

            # Вычисляем средние значения
            mean_h = np.mean(dirty_h)
            mean_s = np.mean(dirty_s)
            mean_v = np.mean(dirty_v)

            # Вычисляем стандартные отклонения
            std_h = np.std(dirty_h)
            std_s = np.std(dirty_s)
            std_v = np.std(dirty_v)

            # КОМБИНИРОВАННЫЙ ПОДХОД: используем percentiles как основу, 
            # но расширяем на основе стандартного отклонения

            # Базовые диапазоны из percentiles
            h_min_base = max(0, int(h_low))
            h_max_base = min(179, int(h_high))
            s_min_base = max(0, int(s_low))
            s_max_base = min(255, int(s_high))
            v_min_base = max(0, int(v_low))
            v_max_base = min(255, int(v_high))

            # Расширяем диапазоны с учетом стандартного отклонения
            tolerance_h = max(5, min(15, int(std_h * 1.5)))  # Запас 5-15 для H
            tolerance_s = max(10, min(30, int(std_s * 1.5))) # Запас 10-30 для S
            tolerance_v = max(10, min(30, int(std_v * 1.5))) # Запас 10-30 для V

            # Применяем зазор
            h_min = max(0, h_min_base - tolerance_h)
            h_max = min(179, h_max_base + tolerance_h)
            s_min = max(0, s_min_base - tolerance_s)
            s_max = min(255, s_max_base + tolerance_s)
            v_min = max(0, v_min_base - tolerance_v)
            v_max = min(255, v_max_base + tolerance_v)

            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: убеждаемся, что диапазон достаточно широк
            min_h_range = 10
            min_s_range = 20
            min_v_range = 20

            if (h_max - h_min) < min_h_range:
                center_h = (h_min + h_max) // 2
                h_min = max(0, center_h - min_h_range // 2)
                h_max = min(179, center_h + min_h_range // 2)

            if (s_max - s_min) < min_s_range:
                center_s = (s_min + s_max) // 2
                s_min = max(50, center_s - min_s_range // 2)  # Минимум насыщенности 50
                s_max = min(255, center_s + min_s_range // 2)

            if (v_max - v_min) < min_v_range:
                center_v = (v_min + v_max) // 2
                v_min = max(50, center_v - min_v_range // 2)  # Минимум яркости 50
                v_max = min(255, center_v + min_v_range // 2)

            # ФИНАЛЬНАЯ ПРОВЕРКА корректности диапазона
            if h_min >= h_max or s_min >= s_max or v_min >= v_max:
                # Используем fallback значения если диапазон некорректен
                h_min, h_max = 35, 85
                s_min, s_max = 50, 255
                v_min, v_max = 50, 255
                thread_safe_logger.log('WARNING', "Автоматическая калибровка не удалась, используются значения по умолчанию")

            # Логирование результатов калибровки
            calibration_info = (
                f"Калибровка завершена:\n"
                f"H: {h_min}-{h_max} (диапазон: {h_max-h_min})\n"
                f"S: {s_min}-{s_max} (диапазон: {s_max-s_min})\n"
                f"V: {v_min}-{v_max} (диапазон: {v_max-v_min})\n"
                f"Использовано пикселей: чистая область - {len(clean_h)}, загрязненная - {len(dirty_h)}"
            )
            thread_safe_logger.log('INFO', calibration_info)

            # Показываем информацию о калибровке
            messagebox.showinfo("✅ Калибровка завершена", 
                              f"Диапазоны HSV определены:\n\n"
                              f"Hue: {h_min} - {h_max}\n"
                              f"Saturation: {s_min} - {s_max}\n"
                              f"Value: {v_min} - {v_max}")

            # Возвращаем результат
            self.callback((h_min, h_max, s_min, s_max, v_min, v_max))
            self.window.destroy()

        except Exception as e:
            error_msg = f"Ошибка при калибровке: {str(e)}"
            thread_safe_logger.log('ERROR', f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка калибровки", 
                               f"{error_msg}\n\n"
                               f"Попробуйте выбрать другие области или проверьте изображение.")

class EnhancedRadioactiveDustAnalyzer:
    """Улучшенный анализатор с точным алгоритмом выделения цвета"""
    def __init__(self, root):
        self.root = root
        self.theme = ModernTheme()
        self.advanced_processor = AdvancedImageProcessor()
        self.area_calculator = AreaCalculator()
        self.image_cache = ImageCache()

        thread_safe_logger.set_root(self.root)
        self.setup_variables()
        self.setup_window()
        self.setup_styles()
        self.setup_ui()
        self.setup_bindings()

    def setup_variables(self):
        """Инициализация переменных"""
        self.hsv_lower = np.array([35, 50, 50])
        self.hsv_upper = np.array([85, 255, 255])
        self.selected_color = (0, 255, 0)
        self.hsv_color = np.array([60, 255, 255])

        self.image = None
        self.mask = None
        self.current_image_path = None
        self.original_image_for_display = None

        self.scale_mm_per_pixel = None

        self.width_var = tk.DoubleVar(value=100.0)
        self.height_var = tk.DoubleVar(value=100.0)

        self.min_area_var = tk.IntVar(value=100)
        self.kernel_size_var = tk.IntVar(value=5)
        self.contour_color_var = tk.StringVar(value="red")

        self.slider_vars = {}

    def setup_window(self):
        """Настройка главного окна"""
        self.root.title("🔬 Анализатор загрязнений")

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1600, screen_width - 100)
        window_height = min(900, screen_height - 100)
        self.root.geometry(f"{window_width}x{window_height}")

        # Разрешаем изменение размеров окна
        self.root.minsize(1200, 700)
        self.root.configure(bg=self.theme.COLORS['background'])

        # Разрешаем изменение размеров всех фреймов
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def setup_styles(self):
        """Настройка стилей элементов"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('.', background=self.theme.COLORS['background'])
        style.configure('Primary.TFrame', background=self.theme.COLORS['background'])
        style.configure('Secondary.TFrame', background=self.theme.COLORS['primary'])
        style.configure('Panel.TFrame', background=self.theme.COLORS['panel'])
        style.configure('Title.TLabel', font=self.theme.FONTS['title'],
                       background=self.theme.COLORS['background'],
                       foreground=self.theme.COLORS['primary'])
        style.configure('Heading.TLabel', font=self.theme.FONTS['heading'],
                       background=self.theme.COLORS['background'],
                       foreground=self.theme.COLORS['text'])
        style.configure('Normal.TLabel', font=self.theme.FONTS['normal'],
                       background=self.theme.COLORS['background'])
        style.configure('Small.TLabel', font=self.theme.FONTS['small'],
                       background=self.theme.COLORS['background'])
        style.configure('Primary.TButton', font=self.theme.FONTS['button'])
        style.configure('Success.TButton', font=self.theme.FONTS['button'])
        style.configure('TLabelframe', background=self.theme.COLORS['background'])
        style.configure('TLabelframe.Label', font=self.theme.FONTS['heading'],
                       background=self.theme.COLORS['background'],
                       foreground=self.theme.COLORS['primary'])

    def setup_bindings(self):
        """Настройка горячих клавиш"""
        bindings = {
            "<Control-o>": lambda e: self.load_image(),
            "<Control-r>": lambda e: self.analyze(),
            "<Control-s>": lambda e: self.save_results(),
            "<Control-c>": lambda e: self.copy_to_clipboard(),
            "<Control-l>": lambda e: self.open_ruler(),
            "<Control-p>": lambda e: self.open_color_calibration(),
            "<F1>": lambda e: self.show_instructions(),
            "<Escape>": lambda e: self.reset_analysis()
        }
        for key, command in bindings.items():
            self.root.bind(key, command)

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        main_container = ttk.Frame(self.root, style='Primary.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Разрешаем изменение размеров главного контейнера
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)

        self.create_header(main_container)

        content_frame = ttk.Frame(main_container, style='Primary.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Разрешаем изменение размеров контентной области
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)

        # Левая панель управления - теперь шире
        left_panel = self.create_control_panel(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Правая панель отображения
        right_panel = self.create_display_panel(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_header(self, parent):
        """Создание заголовка приложения"""
        header_frame = ttk.Frame(parent, style='Primary.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame,
                               text="🔬 Анализатор загрязнений",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT, padx=15, pady=10)

        button_frame = ttk.Frame(header_frame, style='Primary.TFrame')
        button_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        buttons = [
            ("📖 Инструкция (F1)", self.show_instructions),
            ("🎨 Калибровка цвета (Ctrl+P)", self.open_color_calibration),
            ("📏 Линейка (Ctrl+L)", self.open_ruler),
            ("📊 Пакетный анализ", self.open_batch_analysis),  # Добавлена кнопка пакетного анализа
        ]
        for text, command in buttons:
            btn = ttk.Button(button_frame, text=text, command=command, width=20)
            btn.pack(side=tk.LEFT, padx=3)

    def create_control_panel(self, parent):
        """Создание расширенной панели управления"""
        control_frame = ttk.Frame(parent, width=550)  # Увеличили ширину
        control_frame.pack(fill=tk.BOTH, expand=False)
        control_frame.pack_propagate(False)

        canvas = tk.Canvas(control_frame, highlightthickness=0, bg=self.theme.COLORS['background'])
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=530)  # Увеличили ширину
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Секция загрузки изображения
        load_frame = ttk.LabelFrame(scrollable_frame, text="📁 Загрузка изображения", padding=10)
        load_frame.pack(fill=tk.X, pady=5, padx=5)
        self.create_load_section(load_frame)

        # Секция размеров пластины
        size_frame = ttk.LabelFrame(scrollable_frame, text="📐 Параметры пластины", padding=10)
        size_frame.pack(fill=tk.X, pady=5, padx=5)
        self.create_size_controls(size_frame)

        # Секция выбора цвета
        color_frame = ttk.LabelFrame(scrollable_frame, text="🎨 Выбор цвета загрязнения", padding=10)
        color_frame.pack(fill=tk.X, pady=5, padx=5)
        self.create_color_section(color_frame)

        # Секция параметров обработки
        processing_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Параметры обработки", padding=10)
        processing_frame.pack(fill=tk.X, pady=5, padx=5)
        self.create_processing_controls(processing_frame)

        # Секция HSV слайдеров - теперь расширенная
        hsv_frame = ttk.LabelFrame(scrollable_frame, text="🌈 Точная настройка HSV", padding=10)
        hsv_frame.pack(fill=tk.X, pady=5, padx=5)
        self.create_hsv_controls(hsv_frame)

        # Компактный визуализатор диапазона HSV - теперь шире
        self.range_visualizer = CompactHSVRangeVisualizer(scrollable_frame, width=520)

        # Кнопки управления
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=5)

        ttk.Button(button_frame, text="🔄 Сбросить HSV",
                  command=self.reset_hsv_sliders).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="🧹 Сбросить анализ",
                  command=self.reset_analysis).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Основная кнопка анализа
        analyze_frame = ttk.Frame(scrollable_frame)
        analyze_frame.pack(fill=tk.X, pady=10, padx=5)

        analyze_button = ttk.Button(analyze_frame, text="🔍 ВЫПОЛНИТЬ АНАЛИЗ (Ctrl+R)",
                                  command=self.analyze, style='Success.TButton')
        analyze_button.pack(fill=tk.X, pady=5)

        return control_frame

    def create_size_controls(self, parent):
        """Создание элементов управления размерами"""
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.X, pady=5)

        width_frame = ttk.Frame(grid_frame)
        width_frame.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(width_frame, text="Ширина (мм):", style='Normal.TLabel').pack(anchor=tk.W)
        width_entry = ttk.Entry(width_frame, textvariable=self.width_var, width=10, font=self.theme.FONTS['normal'])
        width_entry.pack(pady=2)

        height_frame = ttk.Frame(grid_frame)
        height_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(height_frame, text="Высота (мм):", style='Normal.TLabel').pack(anchor=tk.W)
        height_entry = ttk.Entry(height_frame, textvariable=self.height_var, width=10, font=self.theme.FONTS['normal'])
        height_entry.pack(pady=2)

        calib_frame = ttk.Frame(parent)
        calib_frame.pack(fill=tk.X, pady=10)

        ttk.Label(calib_frame, text="🔧 Калибровка:", style='Normal.TLabel').pack(side=tk.LEFT)
        self.calib_status = ttk.Label(calib_frame, text="Не выполнена", foreground="red", style='Normal.TLabel')
        self.calib_status.pack(side=tk.LEFT, padx=5)
        ttk.Button(calib_frame, text="Настроить", command=self.open_ruler).pack(side=tk.RIGHT)

    def create_load_section(self, parent):
        """Создание секции загрузки"""
        ttk.Button(parent, text="📁 ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ (Ctrl+O)",
                  command=self.load_image, style='Primary.TButton').pack(fill=tk.X, pady=5)

        file_info_frame = ttk.Frame(parent)
        file_info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(file_info_frame, text="📄 Текущий файл:", style='Normal.TLabel').pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_info_frame, text="Файл не выбран", style='Small.TLabel', foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def create_color_section(self, parent):
        """Создание секции выбора цвета"""
        color_display_frame = ttk.Frame(parent)
        color_display_frame.pack(fill=tk.X, pady=10)

        ttk.Label(color_display_frame, text="🎨 Текущий цвет загрязнения:", style='Normal.TLabel').pack(anchor=tk.W)

        color_info_frame = ttk.Frame(color_display_frame)
        color_info_frame.pack(fill=tk.X, pady=5)

        self.current_color_display = tk.Label(color_info_frame, text="Не выбран",
                                             relief='solid', borderwidth=2,
                                             width=15, height=2, font=self.theme.FONTS['normal'])
        self.current_color_display.pack(side=tk.LEFT, padx=(0, 10))

        self.color_info_label = ttk.Label(color_info_frame, text="RGB: ---\nHSV: ---",
                                         style='Small.TLabel', justify=tk.LEFT)
        self.color_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        color_buttons_frame = ttk.Frame(parent)
        color_buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(color_buttons_frame, text="🎨 Выбрать из палитры",
                  command=self.open_color_picker).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(color_buttons_frame, text="🖱️ Калибровка по областям",
                  command=self.open_color_calibration).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def create_processing_controls(self, parent):
        """Создание элементов управления обработкой"""
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.X, pady=5)

        ttk.Label(grid_frame, text="📏 Мин. площадь:", style='Normal.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        min_area_entry = ttk.Entry(grid_frame, textvariable=self.min_area_var, width=8, font=self.theme.FONTS['normal'])
        min_area_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(grid_frame, text="🔧 Размер ядра:", style='Normal.TLabel').grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        kernel_size_entry = ttk.Entry(grid_frame, textvariable=self.kernel_size_var, width=8, font=self.theme.FONTS['normal'])
        kernel_size_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(grid_frame, text="🎨 Цвет выделения:", style='Normal.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        color_combo = ttk.Combobox(grid_frame, textvariable=self.contour_color_var,
                                  values=["red", "green", "blue", "yellow", "cyan", "magenta"],
                                  width=8, state="readonly", font=self.theme.FONTS['normal'])
        color_combo.set("red")
        color_combo.grid(row=1, column=1, padx=5, pady=5)

    def create_hsv_controls(self, parent):
        """Создание расширенных элементов управления HSV"""
        sliders_config = [
            ("H_min", "Hue Min", 0, 179),
            ("H_max", "Hue Max", 0, 179),
            ("S_min", "Saturation Min", 0, 255),
            ("S_max", "Saturation Max", 0, 255),
            ("V_min", "Value Min", 0, 255),
            ("V_max", "Value Max", 0, 255)
        ]
        default_values = {'H_min': 35, 'H_max': 85, 'S_min': 50, 'S_max': 255, 'V_min': 50, 'V_max': 255}

        for i, (var_name, label, min_val, max_val) in enumerate(sliders_config):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=5)

            label_frame = ttk.Frame(frame)
            label_frame.pack(fill=tk.X)

            ttk.Label(label_frame, text=label, style='Normal.TLabel', width=15).pack(side=tk.LEFT)  # Увеличили ширину

            var = tk.IntVar(value=default_values[var_name])
            self.slider_vars[var_name] = var
            value_label = ttk.Label(label_frame, textvariable=var, width=4, style='Normal.TLabel',
                                  background='white', relief='solid', anchor=tk.CENTER)
            value_label.pack(side=tk.RIGHT, padx=5)

            # Увеличили длину слайдера
            slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                              orient=tk.HORIZONTAL, length=450,  # Увеличили длину
                              command=lambda v, p=var_name: self.on_slider_change())
            slider.pack(fill=tk.X, expand=True, padx=5)

    def create_display_panel(self, parent):
        """Создание панели отображения"""
        display_frame = ttk.Frame(parent, style='Primary.TFrame')

        # Разрешаем изменение размеров панели отображения
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        notebook = ttk.Notebook(display_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        images_tab = ttk.Frame(notebook)
        notebook.add(images_tab, text="📷 Изображения")
        self.setup_images_tab(images_tab)

        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="📊 Результаты")
        self.setup_results_tab(results_tab)

        return display_frame

    def setup_images_tab(self, parent):
        """Настройка вкладки изображений"""
        images_frame = ttk.Frame(parent)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Разрешаем изменение размеров фрейма изображений
        images_frame.grid_rowconfigure(0, weight=1)
        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)

        orig_frame = ttk.LabelFrame(images_frame, text="Исходное изображение")
        orig_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Разрешаем изменение размеров фрейма оригинального изображения
        orig_frame.grid_rowconfigure(0, weight=1)
        orig_frame.grid_columnconfigure(0, weight=1)

        self.original_label = tk.Label(orig_frame, background='white',
                                      anchor=tk.CENTER, cursor="arrow",
                                      text="Изображение не загружено")
        self.original_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        mask_frame = ttk.LabelFrame(images_frame, text="Маска загрязнения")
        mask_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Разрешаем изменение размеров фрейма маски
        mask_frame.grid_rowconfigure(0, weight=1)
        mask_frame.grid_columnconfigure(0, weight=1)

        self.mask_label = tk.Label(mask_frame, background='white', anchor=tk.CENTER,
                                  text="Маска не сгенерирована")
        self.mask_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Инициализация менеджера отображения
        self.image_display_manager = ImageDisplayManager(self.original_label, self.mask_label)

    def setup_results_tab(self, parent):
        """Настройка вкладки результатов"""
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Разрешаем изменение размеров фрейма результатов
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        stats_frame = ttk.LabelFrame(results_frame, text="📊 Результаты анализа", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Разрешаем изменение размеров текстового поля
        stats_frame.grid_rowconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(0, weight=1)

        self.stats_text = tk.Text(stats_frame, height=15, wrap=tk.WORD,
                                 font=('Consolas', 10), undo=True)
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="💾 Сохранить результаты (Ctrl+S)",
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📋 Копировать в буфер (Ctrl+C)",
                  command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🖼️ Экспорт изображения",
                  command=self.export_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🔄 Сбросить все",
                  command=self.reset_analysis).pack(side=tk.RIGHT, padx=5)

    # ОСНОВНЫЕ ФУНКЦИИ ПРОГРАММЫ
    def show_instructions(self):
        """Показать инструкцию по использованию программы"""
        instructions = """
📖 ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ
1. ЗАГРУЗКА ИЗОБРАЖЕНИЯ (Ctrl+O)
   - Нажмите "📁 ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ"
   - Поддерживаются форматы: JPG, PNG, BMP, TIFF
2. ВЫБОР ЦВЕТА ЗАГРЯЗНЕНИЯ
   - 🎨 Из палитры: нажмите "Выбрать из палитры"
   - 🖱️ По областям: нажмите "Калибровка по областям" для автоматического определения HSV
3. КАЛИБРОВКА МАСШТАБА
   - Используйте "Линейка" для установки масштаба
4. НАСТРОЙКА ПАРАМЕТРОВ
   - Укажите размеры пластины в мм
   - Настройте диапазон HSV при необходимости
5. АНАЛИЗ (Ctrl+R)
   - Нажмите "🔍 ВЫПОЛНИТЬ АНАЛИЗ"
   - Результаты появятся во вкладке "📊 РЕЗУЛЬТАТЫ"
6. СОХРАНЕНИЕ
   - Сохраните результаты: Ctrl+S
   - Скопируйте в буфер: Ctrl+C
7. ПАКЕТНЫЙ АНАЛИЗ
   - Нажмите "📊 Пакетный анализ" для обработки нескольких изображений

ГОРЯЧИЕ КЛАВИШИ:
  Ctrl+O - Загрузить изображение
  Ctrl+R - Анализ
  Ctrl+S - Сохранить
  Ctrl+C - Копировать
  Ctrl+L - Калибровка линейкой
  Ctrl+P - Калибровка цвета по областям
  F1     - Инструкция
  Esc    - Сброс анализа
        """
        messagebox.showinfo("📖 Инструкция", instructions)

    def open_batch_analysis(self):
        """Открытие инструмента пакетного анализа"""
        if not hasattr(self, 'batch_tool') or not self.batch_tool or not self.batch_tool.window.winfo_exists():
            self.batch_tool = BatchAnalysisTool(self.root, self)
        else:
            self.batch_tool.window.lift()

    def open_color_calibration(self):
        """Открыть инструмент калибровки цвета по областям"""
        if self.image is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите изображение")
            return
        self.color_calibration_tool = ColorCalibrationTool(self.root, self.image, self.apply_calibration_result)

    def apply_calibration_result(self, hsv_range):
        """Применить результаты калибровки"""
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_range

        self.slider_vars['H_min'].set(h_min)
        self.slider_vars['H_max'].set(h_max)
        self.slider_vars['S_min'].set(s_min)
        self.slider_vars['S_max'].set(s_max)
        self.slider_vars['V_min'].set(v_min)
        self.slider_vars['V_max'].set(v_max)

        if hasattr(self, 'range_visualizer'):
            self.range_visualizer.update_range(h_min, h_max, s_min, s_max, v_min, v_max)

        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, s_max, v_max])

        messagebox.showinfo("✅ Успех", "Калибровка цвета применена успешно!")

    def open_ruler(self):
        """Открыть инструмент калибровки линейкой"""
        if self.image is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите изображение")
            return
        self.ruler_tool = RulerTool(self.root, self.image, self.apply_ruler_result)

    def apply_ruler_result(self, scale):
        """Применить результаты калибровки линейкой"""
        self.scale_mm_per_pixel = scale
        self.calib_status.config(text=f"Выполнена ({scale:.4f} мм/пиксель)", foreground="green")
        messagebox.showinfo("✅ Успех", f"Масштаб установлен: {scale:.4f} мм/пиксель")

    def open_color_picker(self):
        """Открыть диалог выбора цвета"""
        color = colorchooser.askcolor(title="Выберите цвет загрязнения")
        if color[0] is not None:
            rgb_color = tuple(int(c) for c in color[0])
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
            self.set_selected_color(rgb_color, hsv_color)

    def set_selected_color(self, rgb_color, hsv_color=None):
        """Установка выбранного цвета"""
        try:
            self.selected_color = rgb_color

            if hsv_color is not None:
                self.hsv_color = hsv_color

            hex_color = f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'
            self.current_color_display.config(
                bg=hex_color, 
                text="",
                foreground='white' if sum(rgb_color) < 380 else 'black'
            )

            if hsv_color is not None:
                color_info = f"RGB: {rgb_color}\nHSV: ({hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]})"
            else:
                color_info = f"RGB: {rgb_color}\nHSV: ---"

            self.color_info_label.config(text=color_info)

            thread_safe_logger.log('INFO', f"Цвет установлен: RGB{rgb_color}")

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка установки цвета: {e}")

    def reset_analysis(self):
        """Сбросить результаты анализа"""
        try:
            self.image = None
            self.mask = None
            self.current_image_path = None
            self.original_image_for_display = None
            self.image_cache.clear()

            self.original_label.configure(image='', text="Изображение не загружено")
            self.mask_label.configure(image='', text="Маска не сгенерирована")
            self.stats_text.delete(1.0, tk.END)
            self.file_label.config(text="Файл не выбран")

            self.current_color_display.config(bg='SystemButtonFace', text="Не выбран")
            self.color_info_label.config(text="RGB: ---\nHSV: ---")

            self.scale_mm_per_pixel = None
            self.calib_status.config(text="Не выполнена", foreground="red")

            thread_safe_logger.log('INFO', "Анализ сброшен")
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка при сбросе анализа: {e}")

    def save_results(self):
        """Сохранить результаты анализа"""
        if self.image is None:
            messagebox.showwarning("Ошибка", "Нет данных для сохранения")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    content = self.stats_text.get(1.0, tk.END)
                    if not content.strip():
                        f.write("Нет данных анализа")
                    else:
                        f.write(content)
                messagebox.showinfo("✅ Успех", "Результаты сохранены")
                thread_safe_logger.log('INFO', f"Результаты сохранены в {filename}")
            except Exception as e:
                thread_safe_logger.log('ERROR', f"Ошибка сохранения результатов: {e}")
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def copy_to_clipboard(self):
        """Копировать результаты в буфер обмена"""
        try:
            content = self.stats_text.get(1.0, tk.END).strip()
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                messagebox.showinfo("✅ Успех", "Результаты скопированы в буфер обмена")
                thread_safe_logger.log('INFO', "Результаты скопированы в буфер обмена")
            else:
                messagebox.showwarning("Ошибка", "Нет данных для копирования")
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка копирования в буфер: {e}")
            messagebox.showerror("Ошибка", f"Не удалось скопировать в буфер: {str(e)}")

    def export_image(self):
        """Экспорт изображения с результатами"""
        if self.image is None:
            messagebox.showwarning("Ошибка", "Нет данных для экспорта")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                      ("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if filename:
            try:
                if hasattr(self, 'mask') and self.mask is not None:
                    result_img = self.image.copy()
                    color_map = {
                        'red': [0, 0, 255],
                        'green': [0, 255, 0],
                        'blue': [255, 0, 0],
                        'yellow': [0, 255, 255],
                        'cyan': [255, 255, 0],
                        'magenta': [255, 0, 255]
                    }
                    highlight_color = color_map.get(self.contour_color_var.get(), [0, 0, 255])
                    result_img[self.mask > 0] = highlight_color
                    cv2.imwrite(filename, result_img)
                else:
                    cv2.imwrite(filename, self.image)
                messagebox.showinfo("✅ Экспорт", "Изображение успешно экспортировано!")
                thread_safe_logger.log('INFO', f"Изображение экспортировано в {filename}")
            except Exception as e:
                thread_safe_logger.log('ERROR', f"Ошибка экспорта изображения: {e}")
                messagebox.showerror("Ошибка", f"Не удалось экспортировать изображение: {str(e)}")

    def load_image(self, filepath=None):
        """Загрузка изображения"""
        try:
            if filepath is None:
                filepath = filedialog.askopenfilename(
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
                )
            if filepath:
                self.image = cv2.imread(filepath)
                if self.image is None:
                    raise ValueError("Не удалось загрузить изображение")

                self.current_image_path = filepath
                self.file_label.config(text=os.path.basename(filepath))

                self.original_image_for_display = self.image.copy()

                # Используем менеджер отображения
                self.image_display_manager.set_original_image(self.image)
                self.mask_label.configure(image='', text="Маска не сгенерирована")
                self.stats_text.delete(1.0, tk.END)
                thread_safe_logger.log('INFO', f"Изображение загружено: {filepath}")
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка загрузки изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def analyze(self):
        """Анализ с точным алгоритмом выделения цвета"""
        if self.image is None:
            messagebox.showwarning("Ошибка", "Загрузите изображение")
            return

        try:
            self.update_hsv_from_sliders()

            if not self.validate_hsv_range():
                return

            # Всегда используем улучшенную предобработку
            processed_img = self.advanced_processor.enhanced_preprocessing(self.image)

            hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            self.mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            if self.mask is None or cv2.countNonZero(self.mask) == 0:
                messagebox.showwarning("Результат", "Не обнаружено областей загрязнения")
                self.image_display_manager.set_mask_image(self.image)
                return

            self.mask = self.advanced_processor.postprocess_mask(
                self.mask,
                min_area=self.min_area_var.get(),
                kernel_size=self.kernel_size_var.get()
            )

            self.visualize_enhanced_results(processed_img)
            self.calculate_statistics()

            thread_safe_logger.log('INFO', "Анализ успешно завершен")

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка анализа: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Анализ не удался:\n{str(e)}")

    def update_hsv_from_sliders(self):
        """Обновление HSV параметров из слайдеров"""
        try:
            self.hsv_lower = np.array([
                self.slider_vars["H_min"].get(),
                self.slider_vars["S_min"].get(),
                self.slider_vars["V_min"].get()
            ])
            self.hsv_upper = np.array([
                self.slider_vars["H_max"].get(),
                self.slider_vars["S_max"].get(),
                self.slider_vars["V_max"].get()
            ])
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка обновления HSV из слайдеров: {e}")

    def validate_hsv_range(self) -> bool:
        """Проверка корректности диапазона HSV"""
        try:
            if (self.hsv_lower[0] >= self.hsv_upper[0] or
                self.hsv_lower[1] >= self.hsv_upper[1] or
                self.hsv_lower[2] >= self.hsv_upper[2]):
                messagebox.showerror("Ошибка", "Некорректный диапазон HSV (min >= max)")
                return False
            return True
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка проверки диапазона HSV: {e}")
            return False

    def on_slider_change(self):
        """Обработчик изменения слайдеров HSV"""
        try:
            h_min = self.slider_vars["H_min"].get()
            h_max = self.slider_vars["H_max"].get()
            s_min = self.slider_vars["S_min"].get()
            s_max = self.slider_vars["S_max"].get()
            v_min = self.slider_vars["V_min"].get()
            v_max = self.slider_vars["V_max"].get()

            if hasattr(self, 'range_visualizer'):
                self.range_visualizer.update_range(h_min, h_max, s_min, s_max, v_min, v_max)

            self.hsv_lower = np.array([h_min, s_min, v_min])
            self.hsv_upper = np.array([h_max, s_max, v_max])
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка обработки изменения слайдеров: {e}")

    def reset_hsv_sliders(self):
        """Сброс слайдеров HSV к значениям по умолчанию"""
        try:
            default_values = {
                'H_min': 35, 'H_max': 85,
                'S_min': 50, 'S_max': 255,
                'V_min': 50, 'V_max': 255
            }
            for slider_name, value in default_values.items():
                self.slider_vars[slider_name].set(value)

            if hasattr(self, 'range_visualizer'):
                self.range_visualizer.update_range(**default_values)

            self.hsv_lower = np.array([default_values['H_min'], default_values['S_min'], default_values['V_min']])
            self.hsv_upper = np.array([default_values['H_max'], default_values['S_max'], default_values['V_max']])

            thread_safe_logger.log('INFO', "Слайдеры HSV сброшены к значениям по умолчанию")

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка сброса слайдеров HSV: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сбросить настройки HSV: {str(e)}")

    def visualize_enhanced_results(self, processed_img: np.ndarray):
        """Визуализация улучшенных результатов с настраиваемым цветом"""
        try:
            overlay = processed_img.copy()

            color_map = {
                'red': [0, 0, 255],
                'green': [0, 255, 0],
                'blue': [255, 0, 0],
                'yellow': [0, 255, 255],
                'cyan': [255, 255, 0],
                'magenta': [255, 0, 255]
            }
            highlight_color = color_map.get(self.contour_color_var.get(), [0, 0, 255])

            colored_mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
            colored_mask[self.mask > 0] = highlight_color

            alpha = 0.6
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

            self.image_display_manager.set_mask_image(overlay)
        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка визуализации результатов: {e}")
            self.image_display_manager.set_mask_image(self.mask)

    def calculate_statistics(self):
        """Расчет и отображение статистики"""
        if self.image is None or self.mask is None:
            return

        try:
            stats = self.area_calculator.calculate_area_stats(
                self.mask,
                self.image.shape[:2],
                self.scale_mm_per_pixel,
                self.width_var.get(),
                self.height_var.get()
            )

            if not stats:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, "Не удалось рассчитать статистику")
                return

            calibration_info = "Выполнена" if self.scale_mm_per_pixel else "Не выполнена"
            scale_info = f"{self.scale_mm_per_pixel:.4f}" if self.scale_mm_per_pixel else "N/A"

            result_text = f"""
📊 РЕЗУЛЬТАТЫ АНАЛИЗА
{'=' * 50}
📐 ПАРАМЕТРЫ ПЛАСТИНЫ:
  • Ширина: {self.width_var.get():.2f} мм
  • Высота: {self.height_var.get():.2f} мм
  • Общая площадь: {stats['total_area_mm2']:.2f} мм²

🔍 СТАТИСТИКА ЗАГРЯЗНЕНИЯ:
  • Площадь загрязнения: {stats['contaminated_mm2']:.2f} мм²
  • Процент загрязнения: {stats['percent']:.2f}%
  • Количество областей: {stats['num_contours']}
  • Пикселей загрязнения: {stats['contaminated_px']:,} из {stats['total_pixels']:,}
  • Площадь крупнейшей области: {stats['largest_contour_area_mm2']:.2f} мм²

📋 ПАРАМЕТРЫ АНАЛИЗА:
  • Предобработка: Всегда включена
  • Диапазон HSV: H({self.hsv_lower[0]}-{self.hsv_upper[0]}) S({self.hsv_lower[1]}-{self.hsv_upper[1]}) V({self.hsv_lower[2]}-{self.hsv_upper[2]})
  • Калибровка: {calibration_info} ({scale_info} мм/пиксель)

⏰ ВРЕМЯ АНАЛИЗА: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, result_text)

            thread_safe_logger.log('INFO', 
                f"Анализ завершен: {stats['contaminated_mm2']:.2f} мм² ({stats['percent']:.2f}%)")

        except Exception as e:
            thread_safe_logger.log('ERROR', f"Ошибка расчета статистики: {e}")
            messagebox.showerror("Ошибка", f"Не удалось рассчитать статистику: {str(e)}")

def main():
    """Главная функция приложения"""
    try:
        root = tk.Tk()
        app = EnhancedRadioactiveDustAnalyzer(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"Критическая ошибка приложения:\n{str(e)}\n{traceback.format_exc()}"
        logging.critical(error_msg)
        messagebox.showerror("Критическая ошибка",
                           f"Приложение завершилось с ошибкой:\n{str(e)}\n"
                           f"Подробности в файле Result.log")

if __name__ == "__main__":
    main()
