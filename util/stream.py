import yt_dlp
import cv2

youtube_url = "https://youtube.com/live/3rwoJ-B8AaM"

ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "force_generic_extractor": True,
    "simulate": True,
    "format": "best",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=False)
    hls_url = info_dict['url']

cap = cv2.VideoCapture(hls_url)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    w, h, _ = frame.shape
    frame = cv2.resize(frame, (w, h))
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
