# Araç Sayma ve Sınıflandırma Projesi (Traffic Counting Project)

Bu proje, **YOLOv8** modelini kullanarak video üzerindeki araçları tespit eder, takip eder ve belirlenen bir referans çizgisini geçen araçları sınıflarına göre (otomobil, kamyon vb.) sayar. Ayrıca, araçların toplam sayısını ve sınıf bazlı dağılımını video üzerine anlık olarak işler ve sonuçları raporlar.

## 🚀 Özellikler

- **Nesne Tespiti ve Takibi**: Ultralytics YOLOv8 kullanarak yüksek doğrulukta araç tespiti ve çoklu nesne takibi (tracking).
- **Sınıf Bazlı Sayım**: Araçları türlerine göre (örn: car, truck, bus) ayırarak sayma.
- **Görselleştirme**: 
  - Araçların etrafında bounding box ve ID gösterimi.
  - Sayım çizgisi ve geçiş efekti.
  - Ekranda anlık sayaç paneli.
- **Video Kaydı**: İşlenen videoyu `.avi` formatında kaydetme.
- **Performans Takibi**: `tqdm` ile işlem ilerlemesini takip etme ve FPS optimizasyonu için frame atlama (frame skip) özelliği.

## 🛠️ Kullanılan Teknolojiler

Bu projede aşağıdaki kütüphaneler ve teknolojiler kullanılmıştır:

- **[Python 3](https://www.python.org/)**: Ana programlama dili.
- **[Ultralytics YOLO](https://docs.ultralytics.com/)**: Nesne tespiti, sınıflandırma ve takip (tracking) için kullanılan derin öğrenme modeli.
- **[OpenCV](https://opencv.org/)**: Görüntü işleme, video okuma/yazma ve çizim işlemleri için.
- **[NumPy](https://numpy.org/)**: Matris ve sayısal işlemler için.
- **[Tqdm](https://github.com/tqdm/tqdm)**: Komut satırında ilerleme çubuğu göstermek için.
- **Lapx**: Tracking algoritmalarının (BoT-SORT, ByteTrack vb.) daha verimli çalışması için kullanılan lineer atama kütüphanesi.

## ⚙️ Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

1. **Gereksinimleri Yükleyin**
   Proje dizininde bir terminal açın ve gerekli Python kütüphanelerini yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Dosyası**
   Proje dizininde eğitilmiş bir YOLO modeli (`best.pt`) veya standart `yolov8n.pt` gibi bir model dosyası bulunmalıdır. Varsayılan olarak kod `best.pt` dosyasını arar.

3. **Video Dosyası**
   İşlenecek videonun (`training_video_1.mp4`) proje dizininde olduğundan emin olun veya `traffic_count.py` dosyasındaki `video_path` değişkenini kendi video yolunuza göre güncelleyin.

## ▶️ Çalıştırma

### 1. Araç Sayma (Inference)
Sistemi çalıştırmak ve videoyu işlemek için:
```bash
python traffic_count.py
```
Bu komut videoyu işler ve sonuçları `sonuc_videosu_sinifli.avi` olarak kaydeder. İşlem tamamlandığında terminalde özet istatistikler gösterilir.

### 2. Model Eğitimi (Opsiyonel)
Eğer kendi veri setinizle modeli yeniden eğitmek isterseniz:
```bash
python model_training.py
```
Bu script, `traffic-flow-counting-j6kxk-21/data.yaml` konumundaki veri setini kullanarak eğitimi başlatır.

## 📂 Dosya Yapısı

- `traffic_count.py`: Ana çalışan script. Video işleme ve sayma mantığı buradadır.
- `model_training.py`: YOLO modelini eğitmek için kullanılan script.
- `requirements.txt`: Proje bağımlılıklarını içeren dosya.
- `best.pt`: Eğitilmiş YOLO ağırlık dosyası.
- `sonuc_videosu_sinifli.avi`: İşlenmiş çıkış videosu.

## 📝 Notlar
- `traffic_count.py` içindeki `line_position` değişkeni ile sayım çizgisinin yerini ayarlayabilirsiniz.
- `FRAME_SKIP` değişkeni ile video işleme hızını artırmak için bazı kareleri atlayabilirsiniz.

---
İyi çalışmalar! 🚀
