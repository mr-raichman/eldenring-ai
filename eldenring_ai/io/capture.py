"""
capture.py - ScreenCapture: streams the game output through wf-recorder into a
v4l2loopback device, reads frames with OpenCV, and returns the greyscale frame
stack that forms the policy's visual observation.
"""

import os
import subprocess
import time
from collections import deque

import cv2
import numpy as np

from eldenring_ai import config
from eldenring_ai.config import paths

V4L2_DEVICE = config.V4L2_DEVICE

WF_RECORDER_CMD = [
    "wf-recorder",
    "-o", config.WAYLAND_OUTPUT,
    "-f", V4L2_DEVICE,
    "--muxer=v4l2",
    "--codec=rawvideo",
    "-x", "bgr24",
]


class ScreenCapture:
    def __init__(self, device: str = V4L2_DEVICE):
        self._device            = device
        self._wf_recorder_proc  = None
        self._frame_buffer = deque(maxlen=config.FRAME_STACK * config.FRAME_SKIP)

        self._ensure_device()
        self._launch_wf_recorder()

        self._cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open capture device {device} even after wf-recorder started. "
                "Try running wf-recorder manually to check for errors."
            )

        total_needed = config.FRAME_STACK * config.FRAME_SKIP
        frames_captured = 0
        for _ in range(total_needed):
            ret, frame = self._cap.read()
            if ret:
                grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grey, (config.OBSERVATION_SHAPE[1], config.OBSERVATION_SHAPE[0]))
                self._frame_buffer.append(resized)
                frames_captured += 1

        if frames_captured == 0:
            raise RuntimeError(
                "wf-recorder is running but no frames could be read from /dev/video0.\n"
                f"Check {paths.WF_LOG} for errors."
            )


    def _ensure_device(self):
        if os.path.exists(self._device):
            return
        result = subprocess.run(
            ["sudo", "modprobe", "v4l2loopback",
             "devices=1", 'card_label="capture"', "exclusive_caps=1"],
            capture_output=True,
        )
        time.sleep(1)

        if not os.path.exists(self._device):
            raise RuntimeError(
                f"Failed to create {self._device} after loading v4l2loopback.\n"
                f"modprobe stderr: {result.stderr.decode().strip()}\n"
                "Try running manually:\n"
                "  sudo modprobe v4l2loopback devices=1 card_label=capture exclusive_caps=1"
            )

    def _launch_wf_recorder(self):
        subprocess.run(["pkill", "-9", "-x", "wf-recorder"], capture_output=True)
        time.sleep(0.5)

        env = os.environ.copy()
        env.setdefault("WAYLAND_DISPLAY", "wayland-1")
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

        log_path = str(paths.WF_LOG)
        self._wf_log = open(log_path, "w")

        self._wf_recorder_proc = subprocess.Popen(
            WF_RECORDER_CMD,
            stdout=subprocess.DEVNULL,
            stderr=self._wf_log,
            env=env,
        )
        time.sleep(2)

        if self._wf_recorder_proc.poll() is not None:
            self._wf_log.flush()
            with open(log_path) as f:
                error = f.read().strip()
            raise RuntimeError(
                f"wf-recorder exited immediately (code {self._wf_recorder_proc.returncode}).\n"
                f"stderr: {error or '(empty)'}\n"
                f"Full log: {log_path}"
            )

    def get_frame(self):
        ret, frame = self._cap.read()

        if not ret or frame is None:
            raise RuntimeError(
                "Failed to read frame from capture device. "
                "wf-recorder may have stopped. Restart training."
            )

        height, width = config.OBSERVATION_SHAPE[0], config.OBSERVATION_SHAPE[1]

        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grey, (width, height))

        self._frame_buffer.append(resized)

        frames = list(self._frame_buffer)

        sampled = []
        for i in range(config.FRAME_STACK):
            idx = min(i * config.FRAME_SKIP + 1, len(frames))
            sampled.append(frames[-idx])
        sampled.reverse()

        observation = np.stack(sampled, axis=-1)

        return frame, observation

    def close(self):
        self._cap.release()
        if self._wf_recorder_proc is not None:
            self._wf_recorder_proc.terminate()
            self._wf_recorder_proc.wait()
        if hasattr(self, "_wf_log"):
            self._wf_log.close()
