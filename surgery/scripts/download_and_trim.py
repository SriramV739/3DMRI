import os
import cv2
import yt_dlp
from pathlib import Path

def download_and_trim(video_id: str, output_path: str, start_sec: int = 60, duration_sec: int = 30):
    raw_video_path = f"raw_{video_id}.mp4"

    # Download the video using yt-dlp (max 720p to save space)
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': raw_video_path,
        'quiet': False,
    }

    print(f"[info] Downloading {video_id}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    if not os.path.exists(raw_video_path):
        print(f"[error] Failed to download {video_id}")
        return

    print(f"[info] Trimming {video_id} (Start: {start_sec}s, Duration: {duration_sec}s)...")

    # Use OpenCV to trim the video
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open {raw_video_path} with OpenCV")
        os.remove(raw_video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    max_frames = int(duration_sec * fps)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # We use mp4v codec for standard OpenCV output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Skip to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    while frames_written < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        if frames_written % int(fps * 5) == 0:
            print(f"  Processed {frames_written}/{max_frames} frames...")

    cap.release()
    out.release()

    # Delete the large raw video to save storage!
    print(f"[info] Deleting raw video {raw_video_path}...")
    os.remove(raw_video_path)

    print(f"[success] Saved 30-second clip to {output_path}")

if __name__ == "__main__":
    videos = [
        ("gp11IssuXVs", "data/clip_demo/cholec_clip_1.mp4"),
        ("qbix5XXM0J8", "data/clip_demo/cholec_clip_2.mp4")
    ]

    for vid, out_path in videos:
        download_and_trim(vid, out_path, start_sec=60, duration_sec=30)
