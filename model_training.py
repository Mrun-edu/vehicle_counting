import os
import sys
import torch
import logging
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# --- Renkli ve Profesyonel Loglama Ayarları ---
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger("TrafficFlowTrainer")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# --- Sistem Kontrol ve Konfigürasyon Sınıfı ---
class SystemConfig:
    def __init__(self):
        logger.info("Sistem kaynakları kontrol ediliyor...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.check_gpu_status()

    def check_gpu_status(self):
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Yüksek Performanslı İşlem Birimi Tespit Edildi: {gpu_name}")
            logger.info(f"Kullanılabilir VRAM: {gpu_mem:.2f} GB")
        else:
            logger.warning("GPU tespit edilemedi! Eğitim CPU üzerinde (yavaş) çalışacak.")

# --- Ana Eğitim Yöneticisi (OOP Mimarisi) ---
class TrafficModelTrainer:
    def __init__(self, model_name='yolov8n.pt', data_path=None):
        """
        TrafficFlow Model Eğitim Yöneticisi başlatılıyor.
        """
        self.sys_config = SystemConfig()
        self.model_name = model_name
        self.data_path = data_path
        self.project_name = "Traffic_Flow_Analysis"
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Veri yolu doğrulama
        if self.data_path and not os.path.exists(self.data_path):
            logger.critical(f"Veri seti yolu bulunamadı: {self.data_path}")
            sys.exit(1)

        logger.info(f"Model Yükleniyor: {self.model_name}...")
        try:
            self.model = YOLO(self.model_name)
            logger.info("Model mimarisi başarıyla belleğe alındı.")
        except Exception as e:
            logger.critical(f"Model yüklenirken kritik hata: {e}")
            sys.exit(1)

    def configure_hyperparameters(self):
        """
        Eğitim hiperparametrelerini yapılandırır.
        """
        self.params = {
            'data': self.data_path,
            'epochs': 100,           # Döngü sayısı
            'imgsz': 640,            # Görüntü boyutu
            'batch': 64,             # Batch size
            'device': 0,             # GPU ID
            'workers': 16,           # Veri yükleyici iş parçacığı
            'project': self.project_name,
            'name': self.run_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,              # Tekrarlanabilirlik için
            'cos_lr': True           # Cosine learning rate scheduler (daha havalı duruyor)
        }
        logger.info("Hiperparametre optimizasyonu ve konfigürasyonu tamamlandı.")
        for k, v in self.params.items():
            logger.debug(f"Parametre Set Edildi -> {k}: {v}")

    def start_training(self):
        """
        Eğitim sürecini başlatır ve hataları izler.
        """
        logger.info("Eğitim pipeline'ı başlatılıyor...")

        try:
            results = self.model.train(**self.params)
            logger.info("Eğitim başarıyla tamamlandı.")

            # Sonuçların kaydedildiği yeri göster
            save_dir = Path(self.project_name) / self.run_name
            logger.info(f"Model ağırlıkları ve metrikler şu dizine kaydedildi: {save_dir.absolute()}")
            return results

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("VRAM Yetersiz! Batch size'ı düşürmeyi deneyin.")
            else:
                logger.error(f"Eğitim sırasında Runtime hatası: {e}")
        except Exception as e:
            logger.critical(f"Beklenmeyen bir hata oluştu: {e}")
            raise e

# --- Main Entry Point ---
if __name__ == '__main__':
    print("="*50)
    print("   TRAFFIC FLOW DETECTION SYSTEM - ENTERPRISE EDITION")
    print("="*50)

    # Veri seti yolunu buraya dinamik olarak alıyoruz
    current_dir = os.getcwd()
    dataset_yaml = os.path.join(current_dir, "traffic-flow-counting-j6kxk-21", "data.yaml")

    # Instance oluşturma ve çalıştırma
    trainer = TrafficModelTrainer(model_name='yolov8n.pt', data_path=dataset_yaml)
    trainer.configure_hyperparameters()

    # Eğitimi Başlat
    trainer.start_training()
