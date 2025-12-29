import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# 1. Model ve Video Ayarları
model = YOLO('best.pt')
video_path = "training_video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Video özelliklerini al
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Frame atlama ayarı (Hızlandırma için)
FRAME_SKIP = 1
output_fps = max(1, int(fps / FRAME_SKIP))

out = cv2.VideoWriter('sonuc_videosu_sinifli.avi', cv2.VideoWriter_fourcc(*'MJPG'), output_fps, (w, h))

# 2. Sayma Çizgisi Ayarları
line_position = int(h * 0.82)
offset = 10

# Sayaçlar
total_counter = 0
counted_ids = []
class_counts = {} # YENİ: Sınıf bazlı sayıları tutacak sözlük (Örn: {'car': 5, 'truck': 2})

print(f"Video işleniyor... Toplam frame: {total_frames}. Frame skip: {FRAME_SKIP}")

# Progress bar oluştur
pbar = tqdm(total=total_frames, unit="frame")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    pbar.update(1)

    # Frame atlama mantığı
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        names = model.names

        for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = names[cls_idx] # Aracın sınıf ismi (car, truck vb.)

            # --- SAYMA MANTIĞI ---
            if (line_position - offset) < cy < (line_position + offset):
                if track_id not in counted_ids:
                    # 1. Toplam sayacı artır
                    total_counter += 1
                    counted_ids.append(track_id)

                    # 2. YENİ: Sınıf bazlı sayacı artır
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1 # İlk kez görülüyorsa 1 yap

                    # Çizgi görsel efekti
                    cv2.line(frame, (0, line_position), (w, line_position), (0, 255, 0), 3)

            # Görselleştirme (Kutu ve etiket)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name}-{track_id}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- BİLGİ EKRANI ---
    # Sabit referans çizgisi
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 2)

    # Arka plan için siyah bir kutu (yazılar okunsun diye)
    # Sınıf sayısı arttıkça kutuyu büyütmek gerekebilir, şimdilik sabit verdim
    cv2.rectangle(frame, (0, 0), (250, 100 + (len(class_counts) * 30)), (0, 0, 0), -1)

    # Toplam Sayı
    cv2.putText(frame, f"Toplam: {total_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # YENİ: Sınıf Bazlı Sayıları Ekrana Yazdır
    y_pos = 70 # İlk sınıfın yazılacağı Y koordinatı
    for cls_name, count in class_counts.items():
        text = f"{cls_name}: {count}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30 # Her satırda aşağı kay

    out.write(frame)

pbar.close()
cap.release()
out.release()

# Sonuçları Terminale de yazdıralım
print(f"\nİşlem tamamlandı!")
print(f"Toplam Araç: {total_counter}")
print("Detaylı Döküm:")
for k, v in class_counts.items():
    print(f"- {k}: {v}")