import yt_dlp
import cv2

def captureFrame(youtubeUrl):
    ydlOpts = {
        "quiet": True,
        "skip_download": True,
        "force_generic_extractor": True,
        "simulate": True,
        "format": "best",
    }

    with yt_dlp.YoutubeDL(ydlOpts) as ydl:
        info_dict = ydl.extract_info(youtubeUrl, download=False)
        hls_url = info_dict['url']

    cap = cv2.VideoCapture(hls_url)

    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    w, h, _ = frame.shape
    frame = cv2.resize(frame, (w, h))
    return frame
