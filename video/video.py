import cv2
import time
import os
from moviepy.editor import VideoFileClip
from os.path import join, isfile


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.store_images_path = "./youtube_video/best_images/"

    @staticmethod
    def get_fps(video_path):
        clip = VideoFileClip(video_path)
        fps = round(clip.fps)
        print(f"FPS for video: {fps}")
        return fps
    @staticmethod
    def frame_count(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("FRAME COUNT:", frame_count)
        return frame_count, cap

    @classmethod
    def existing_number_of_images(cls, folder_path):
        files = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
        # Filter only image files (you can customize this based on your image file extensions)
        image_files = [
            f
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
        # Get the number of image files
        number_of_images = len(image_files)
        return number_of_images

    def clearest(self, interval_seconds=3):
        fps = self.get_fps(self.video_path)
        frame_count, cap = self.frame_count(self.video_path)
        number_existing_images = VideoProcessor.existing_number_of_images(self.store_images_path)
        interval_frames = interval_seconds * fps
        for i in range(int(frame_count / interval_frames)):
            arr_frame = []
            arr_lap = []
            for j in range(interval_frames):
                success, frame = cap.read()
                laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()
                arr_lap.append(laplacian)
                arr_frame.append(frame)
            selected_frame = arr_frame[arr_lap.index(max(arr_lap))]
            cv2.imwrite(f"{self.store_images_path}img-{i+number_existing_images}.jpg", selected_frame)


if __name__ == "__main__":
    start = time.time()
    for i in range(1,9):
        vid = VideoProcessor(f"./youtube_video/video_high_split/part_{i}.mp4")
        vid.clearest(3)
    print(time.time() - start)
