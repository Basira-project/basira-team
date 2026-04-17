import os
import time
import cv2
import torch
import sqlite3
import easyocr
import json
import pyaudio
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from vosk import Model, KaldiRecognizer
from datetime import datetime
from torch.nn.functional import cosine_similarity
import re
import keyboard
from gtts import gTTS
from playsound import playsound
from openai import OpenAI
import sounddevice as sd
import time

# =====================================================
# ELM (Arabic TTS)
# =====================================================
client = OpenAI(
    api_key="sk-u9RH-lzA67c60lrdzkHVLw",
    base_url="https://elmodels.ngrok.app/v1"
)

def speak_arabic(text):
    print(text)

    response = client.audio.speech.create(
        model="elm-tts",
        voice="default",
        input=text
    )

    filename = f"temp_{int(time.time())}.wav"

    with open(filename, "wb") as f:
        f.write(response.read())

    from playsound import playsound
    playsound(filename)

    #os.remove(filename)

# =====================================================
# English TTS (gTTS)
# =====================================================
def speak_english(text):
    print(text)
    tts = gTTS(text=text, lang='en')
    filename = "voice.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# =====================================================
# Startup
# =====================================================
speak_arabic("يرجى الانتظار، جاري تحميل النظام")

# =====================================================
# Models
# =====================================================
embedding_model = models.mobilenet_v2(pretrained=True)
embedding_model.classifier = torch.nn.Identity()
embedding_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

reader = easyocr.Reader(['ar', 'en'])

vosk_model = Model("vosk-model-ar-0.22-linto-1.1.0")

conn = sqlite3.connect(r"C:\Basira\basira.db", check_same_thread=False)
cursor = conn.cursor()

speak_arabic("مرحبًا بك في مشروع بصيرة")

# =====================================================
# Utils
# =====================================================
def is_arabic(text):
    return re.search(r'[\u0600-\u06FF]', text)

def speak_mixed_text(text):
    words = text.split()
    ar, en = [], []

    for w in words:
        if is_arabic(w):
            ar.append(w)
        else:
            en.append(w)

    if ar:
        speak_arabic(" ".join(ar))
    if en:
        speak_english(" ".join(en))

# =====================================================
# Camera
# =====================================================
def capture_image(filename="frame.jpg"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(filename, frame)
        return filename
    return None

# =====================================================
# Beep
# =====================================================
def beep():
    import winsound
    winsound.Beep(1000, 200)

# =====================================================
# Embedding
# =====================================================
def get_embedding(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = embedding_model(img)
    return vec.squeeze().numpy()

# =====================================================
# Personal Matching
# =====================================================
def match_personal(frame):
    query_vec = get_embedding(frame)

    cursor.execute("SELECT name, vector FROM personal_embeddings")
    items = cursor.fetchall()

    best_name = None
    best_score = 0

    for name, blob in items:
        stored_vec = np.frombuffer(blob, dtype=np.float32)
        score = cosine_similarity(
            torch.tensor(query_vec),
            torch.tensor(stored_vec),
            dim=0
        ).item()

        if score > best_score:
            best_score = score
            best_name = name

    if best_score > 0.80:
        return best_name
    return None

# =====================================================
# Translation
# =====================================================
def get_arabic_translation(label):
    cursor.execute(
        "SELECT arabic_label FROM yolo_translations WHERE english_label=?",
        (label,)
    )
    result = cursor.fetchone()
    return result[0] if result else label

# =====================================================
# Detection (Arabic only)
# =====================================================
def run_detection():
    img_path = capture_image()
    frame = cv2.imread(img_path)

    if frame is None:
        speak_arabic("حدث خطأ في التصوير")
        return

    item = match_personal(frame)
    if item:
        speak_arabic(f"تم التعرف على {item} أمامك")
        return

    results = model(frame)
    detections = results.pred[0]
    labels = results.names

    if len(detections) == 0:
        speak_arabic("لم يتم التعرف")
        return

    class_id = int(detections[0][5])
    english = labels[class_id]
    arabic = get_arabic_translation(english)

    speak_arabic(f"تم التعرف على {arabic} أمامك")

# =====================================================
# OCR (Arabic + English)
# =====================================================
def run_ocr():
    img_path = capture_image("ocr.jpg")
    img = cv2.imread(img_path)

    results = reader.readtext(img)

    text = " ".join([t for (_, t, _) in results])

    if text:
        print(text)
        speak_mixed_text(text)
    else:
        speak_arabic("لم أتعرف على نص")

# =====================================================
# Register Object
# =====================================================
def register_object():
    speak_arabic("ما اسم العنصر؟")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)
    stream.start_stream()

    recognizer = KaldiRecognizer(vosk_model, 16000)

    object_name = None

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            object_name = result.get("text", "").strip()
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not object_name:
        speak_arabic("لم أسمع الاسم")
        return

    speak_arabic("جاري تسجيل العنصر")

    embeddings = []

    for i in range(30):
        img_path = f"temp_{i}.jpg"
        capture_image(img_path)

        beep()

        frame = cv2.imread(img_path)
        vec = get_embedding(frame)
        embeddings.append(vec)

        time.sleep(0.5)

    avg_vec = np.mean(embeddings, axis=0).astype(np.float32)

    cursor.execute("INSERT OR REPLACE INTO personal_embeddings VALUES (?,?)",
                   (object_name, avg_vec.tobytes()))
    conn.commit()

    speak_arabic(f"تم تسجيل العنصر بنجاح")

# =====================================================
# Time
# =====================================================
def speak_time():
    now = datetime.now()

    hour = now.hour
    minute = now.minute

    if hour == 0:
        hour_12 = 12
        period = "صباحًا"
    elif hour < 12:
        hour_12 = hour
        period = "صباحًا"
    elif hour == 12:
        hour_12 = 12
        period = "مساءً"
    else:
        hour_12 = hour - 12
        period = "مساءً"

    cursor.execute("SELECT arabic_text FROM arabic_numbers WHERE number=?", (hour_12,))
    h = cursor.fetchone()
    hour_text = h[0] if h else str(hour_12)

    cursor.execute("SELECT arabic_text FROM arabic_numbers WHERE number=?", (minute,))
    m = cursor.fetchone()
    minute_text = m[0] if m else str(minute)

    if 1 <= minute <= 10:
        minute_word = "دقايق"
    else:
        minute_word = "دقيقة"

    
    if minute == 0:
        speak_arabic(f"الساعة {hour_text} {period}")
    else:
        speak_arabic(f"الساعة {hour_text} و {minute_text} {minute_word} {period}")

# =====================================================
# MAIN LOOP
# =====================================================
print("""
a = Detect
b = OCR
c = Register
d = Time
q = Exit
""")

while True:
    if keyboard.is_pressed('a'):
        run_detection()
        time.sleep(1)

    elif keyboard.is_pressed('b'):
        run_ocr()
        time.sleep(1)

    elif keyboard.is_pressed('c'):
        register_object()
        time.sleep(1)

    elif keyboard.is_pressed('d'):
        speak_time()
        time.sleep(1)

    elif keyboard.is_pressed('q'):
        speak_arabic("إلى اللقاء")
        break