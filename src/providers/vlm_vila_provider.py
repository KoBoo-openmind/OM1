import logging
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import VideoStream

from .singleton import singleton
from pathlib import Path


@singleton
class VLMVilaProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage audio input streaming and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(self, ws_url: str, fps: int = 30, stream_url: Optional[str] = None):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        fps : int
            The fps for the VLM service connection.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        repo_root = self._find_om1_root(Path(__file__))
        models_dir = (repo_root / "models") if repo_root else None
        if models_dir:
            engine_path = str((models_dir / "scrfd_2.5g_640.engine").resolve())
        self.video_stream: VideoStream = VideoStream(
            self.ws_client.send_message, fps=fps, blur_enabled=(engine_path is not None), 
            scrfd_engine=engine_path,
            scrfd_input= "input.1", scrfd_size=640,verbose=True
        )
    

    def _find_om1_root(self, start: Path) -> Path | None:
        """
        Walk up from `start` to find a directory literally named 'OM1'.
        As a fallback, pick the first ancestor that contains a 'models' dir.
        """
        start = start.resolve()
        for p in (start, *start.parents):
            if p.name == "OM1" and (p / "models").is_dir():
                return p

        # Fallback: first ancestor that has a models/ folder
        for p in (start, *start.parents):
            if (p / "models").is_dir():
                return p

        return None

    def register_frame_callback(self, video_callback: Optional[Callable]):
        """
        Register a callback for processing video frames.

        Parameters
        ----------
        video_callback : callable
            The callback function to process video frames.
        """
        self.video_stream.register_frame_callback(video_callback)

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : callable
            The callback function to process VLM results.
        """
        self.ws_client.register_message_callback(message_callback)

    def start(self):
        """
        Start the VLM provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("Vila VLM provider started")

    def stop(self):
        """
        Stop the VLM provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False

        self.video_stream.stop()
        self.ws_client.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()
