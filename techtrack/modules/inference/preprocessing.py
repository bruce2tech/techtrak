import cv2
import numpy as np
import subprocess
from typing import Generator


class Preprocessing:
    """
    Handles video file reading and frame extraction for object detection inference.

    This class reads a video from a file and preprocesses frames before passing them 
    to an object detection module for inference.
    """

    def __init__(self, filename: str, drop_rate: int = 10) -> None:
        """
        Initializes the Preprocessing class.

        :param filename: Path to the video file.
        :param drop_rate: The interval at which frames are selected. For example, 
                          `drop_rate=10` means every 10th frame is retained.
                          
        :ivar self.filename: Stores the video file path.
        :ivar self.drop_rate: Defines how frequently frames are extracted from the video.
        """
        self.filename = filename
        self.drop_rate = drop_rate

    def capture_video(self):
        # Probe width/height using ffprobe (works for files & UDP streams)
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            self.filename
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise ValueError(f"Error probing '{self.filename}': {res.stderr.strip()}")
        width, height = map(int, res.stdout.strip().split(","))

        # 2) Launch ffmpeg subprocess to read from self.filename
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-listen", "1",
            "-i", self.filename,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"fps=30",        # or choose a frame rate
            "pipe:1"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = width * height * 3

        frame_idx = 0
        try:
            while True:
                # read raw bytes for one frame
                raw = p.stdout.read(frame_size)
                if not raw:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))

                # drop frames according to drop_rate
                if frame_idx % self.drop_rate == 0:
                    yield frame
                frame_idx += 1
        finally:
            p.stdout.close()
            p.wait()