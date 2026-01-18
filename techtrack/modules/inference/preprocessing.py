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

    def _is_stream(self) -> bool:
        """Check if the source is a network stream (UDP/TCP) vs a local file."""
        return self.filename.startswith(("udp://", "tcp://", "rtsp://", "http://", "https://"))

    def capture_video(self):
        is_stream = self._is_stream()

        # Probe width/height using ffprobe (works for files & UDP streams)
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            self.filename
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise ValueError(f"Error probing '{self.filename}': {res.stderr.strip()}")
        width, height = map(int, res.stdout.strip().split(","))

        # Build ffmpeg command based on source type
        cmd = ["ffmpeg", "-nostdin"]
        if is_stream:
            cmd.extend(["-listen", "1"])
        cmd.extend([
            "-i", self.filename,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", "fps=30",
            "pipe:1"
        ])
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