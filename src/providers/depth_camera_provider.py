import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs

from providers.singleton import singleton

logger = logging.getLogger(__name__)


@singleton
class DepthCameraProvider:
    """Provider for depth camera collision avoidance using Intel RealSense.

    This provider reads depth data from a downward-facing camera and detects
    obstacles beneath the robot for collision avoidance.
    """

    def __init__(
        self,
        depth_threshold: float = 0.3,  # Minimum safe distance in meters
        fps: int = 30,  # D435i supports 30fps at max resolution
        width: int = 1280,  # D435i max resolution
        height: int = 720,
    ):
        """Initialize the depth camera provider.

        Args:
            depth_threshold: Minimum safe distance to ground/obstacles in meters
            fps: Frames per second for the depth camera
            width: Width of the depth image
            height: Height of the depth image
        """
        self.depth_threshold = depth_threshold
        self.fps = fps
        self.width = width
        self.height = height

        # RealSense pipeline
        self.pipeline = None
        self.config = None
        self.depth_scale = 0.001  # Default value, will be updated from sensor

        # Thread management
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Obstacle detection results
        self._obstacle_detected = False
        self._min_distance = None
        self._depth_frame = None

    def start(self):
        """Start the depth camera provider."""
        if self._running:
            logger.warning("Depth camera provider already running")
            return

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable depth stream
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

        # Start pipeline
        profile = self.pipeline.start(self.config)

        # Get depth sensor
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        logger.info(f"Depth scale: {self.depth_scale}")

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        logger.info("Depth camera provider started successfully")

    def stop(self):
        """Stop the depth camera provider."""
        logger.info("Stopping depth camera provider")
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)

        if self.pipeline:
            self.pipeline.stop()

        logger.info("Depth camera provider stopped")

    def _run(self):
        """Main processing loop running in separate thread."""
        while self._running:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                # Get depth frame
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())

                # Process depth data
                self._process_depth_frame(depth_image)

                # Store frame for external access
                with self._lock:
                    self._depth_frame = depth_image

            except Exception as e:
                logger.error(f"Error in depth processing loop: {e}")
                time.sleep(0.1)

    def _process_depth_frame(self, depth_image: np.ndarray):
        """Process depth frame to detect obstacles.

        Args:
            depth_image: Depth image as numpy array (values in mm)
        """
        # Convert depth values from mm to meters
        depth_m = depth_image * self.depth_scale

        # Filter valid depth values (ignore 0 values which mean no reading)
        valid_depths = depth_m[depth_m > 0]

        if len(valid_depths) > 0:
            # Get minimum distance
            min_distance = np.min(valid_depths)

            # Update detection status
            with self._lock:
                self._obstacle_detected = min_distance < self.depth_threshold
                self._min_distance = min_distance
        else:
            with self._lock:
                self._obstacle_detected = False
                self._min_distance = None

    @property
    def obstacle_detected(self) -> bool:
        """Check if any obstacle is detected below the threshold."""
        with self._lock:
            return self._obstacle_detected

    @property
    def min_distance(self) -> Optional[float]:
        """Get minimum distance to obstacle in meters."""
        with self._lock:
            return self._min_distance

    @property
    def is_running(self) -> bool:
        """Check if the provider is running."""
        return self._running

    @property
    def depth_frame(self) -> Optional[np.ndarray]:
        """Get the latest depth frame."""
        with self._lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Initializing Depth Camera Provider...")
    camera_provider = DepthCameraProvider()

    logger.info("Starting Depth Camera Provider...")
    camera_provider.start()

    if not camera_provider.is_running:
        logger.error("Failed to start Depth Camera Provider. Exiting.")
        exit()

    logger.info(
        "Depth Camera Provider started. Press 'q' in the display window to quit."
    )

    try:
        while True:
            depth_frame_raw = camera_provider.depth_frame

            if depth_frame_raw is not None:
                # Normalize the depth image for display.
                # RealSense depth frames are typically uint16, representing distance in mm.
                # We'll clip to a max depth (e.g., 5 meters = 5000mm) and scale to 0-255 (uint8).
                max_display_depth_mm = 5000.0

                # Ensure frame is float for division, then clip and scale
                depth_frame_normalized = depth_frame_raw.astype(np.float32)
                depth_frame_normalized = np.clip(
                    depth_frame_normalized, 0, max_display_depth_mm
                )
                depth_frame_normalized = (
                    depth_frame_normalized / max_display_depth_mm
                ) * 255.0
                depth_frame_display = depth_frame_normalized.astype(np.uint8)

                cv2.imshow("Depth Feed", depth_frame_display)
            else:
                # Optional: log if no frame, or just wait briefly
                # logger.debug("No depth frame available yet.")
                time.sleep(
                    0.01
                )  # Wait a bit if no frame to reduce CPU usage in tight loop

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed, stopping...")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C), stopping...")
    finally:
        logger.info("Stopping Depth Camera Provider...")
        camera_provider.stop()
        cv2.destroyAllWindows()
        logger.info("Depth Camera Provider stopped and windows closed.")
