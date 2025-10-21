import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from zenoh import ZBytes

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from zenoh_msgs import (
    AIStatusRequest,
    String,
    geometry_msgs,
    nav_msgs,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton

# Nav2 Action Status Codes
status_map = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",  # Only this status re-enables AI mode
    5: "CANCELED",
    6: "ABORTED",
}


@singleton
class UnitreeGo2NavigationProvider:
    """
    Navigation Provider for Unitree Go2 robot.

    This class implements a singleton pattern to manage:
        * Navigation goal publishing to ROS2 Nav2 stack
        * Navigation status monitoring from ROS2 action server
        * Automatic AI mode control based on navigation state

    The provider automatically manages AI mode control during navigation:
    - Disables AI mode when navigation starts (ACCEPTED/EXECUTING status)
    - Re-enables AI mode only on successful navigation completion (SUCCEEDED status)
    - Keeps AI mode disabled on navigation failure/cancellation (CANCELED/ABORTED status)

    Parameters
    ----------
    navigation_status_topic : str, optional
        The ROS2 topic to subscribe for navigation status messages.
        Default: "navigate_to_pose/_action/status"
        Alternative: "navigate_to_pose/_action/feedback" for more detailed updates
    goal_pose_topic : str, optional
        The topic on which to publish goal poses (default is "goal_pose").
    cancel_goal_topic : str, optional
        The topic on which to publish goal cancellations
        (default is "navigate_to_pose/_action/cancel_goal").
    """

    def __init__(
        self,
        navigation_status_topic: str = "navigate_to_pose/_action/status",
        goal_pose_topic: str = "goal_pose",
        cancel_goal_topic: str = "navigate_to_pose/_action/cancel_goal",
    ):
        """
        Initialize the Unitree Go2 Navigation Provider with a specific topic.

        Parameters
        ----------
        navigation_status_topic : str, optional
            The ROS2 topic to subscribe for navigation status messages.
            Default: "navigate_to_pose/_action/status"
            Alternative: "navigate_to_pose/_action/feedback" for more detailed updates
        goal_pose_topic : str, optional
            The topic on which to publish goal poses (default is "goal_pose").
        cancel_goal_topic : str, optional
            The topic on which to publish goal cancellations (default is "navigate_to_pose/_action/cancel_goal").
        """
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.navigation_status_topic = navigation_status_topic
        self.navigation_status = "UNKNOWN"

        self.goal_pose_topic = goal_pose_topic
        self.cancel_goal_topic = cancel_goal_topic

        self.running: bool = False
        self._nav_in_progress: bool = False
        self._current_destination: Optional[str] = None  # Track destination name
        
        # Goal tracking for duplicate prevention
        self._last_published_goal: Optional[geometry_msgs.PoseStamped] = None
        self._last_destination_name: Optional[str] = None
        self._last_goal_time: float = 0.0
        self._goal_publish_count: int = 0
        self._arrival_announced: bool = False  # Prevent duplicate arrival announcements

        # TTS provider for speech feedback
        self.tts_provider = ElevenLabsTTSProvider()

        # AI status control
        self.ai_status_topic = "om/ai/request"
        self.ai_status_pub = None
        if self.session:
            try:
                self.ai_status_pub = self.session.declare_publisher(
                    self.ai_status_topic
                )
                logging.info(
                    "AI status publisher initialized on topic: %s", self.ai_status_topic
                )
            except Exception as e:
                logging.error(f"Error creating AI status publisher: {e}")

    def navigation_status_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming navigation status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            message: nav_msgs.Nav2Status = nav_msgs.Nav2Status.deserialize(
                data.payload.to_bytes()
            )
            logging.debug("Received Navigation Status message: %s", message)
            status_list = message.status_list
            if status_list:
                latest_status = status_list[-1]  # type: ignore
                status_code = latest_status.status
                self.navigation_status = status_map.get(status_code, "UNKNOWN")
                logging.info(
                    "Received navigation status from ROS2 topic '/navigate_to_pose/_action/status': %s (code=%d)",
                    self.navigation_status,
                    status_code,
                )

                # Track navigation state and AI mode control
                # AI mode is ONLY re-enabled on STATUS_SUCCEEDED (4)
                if status_code in (1, 2):  # ACCEPTED or EXECUTING
                    if not self._nav_in_progress:
                        self._nav_in_progress = True
                        self._publish_ai_status(
                            enabled=False
                        )  # Disable AI during navigation
                        logging.info("Navigation started - AI mode disabled")
                elif (
                    status_code == 4
                ):  # STATUS_SUCCEEDED - Navigation completed successfully
                    if self._nav_in_progress:
                        self._nav_in_progress = False
                        self._publish_ai_status(
                            enabled=True
                        )  # Re-enable AI ONLY on success
                        logging.info("Navigation succeeded - AI mode re-enabled")

                        # DON'T clear goal tracking immediately on success
                        # Keep destination name to prevent immediate repeated navigation
                        self._last_published_goal = None  # Clear pose but keep destination name
                        # self._last_destination_name = None  # KEEP this to prevent repeats
                        # self._last_goal_time = 0.0  # KEEP this to prevent repeats
                        
                        logging.info(f"Keeping destination '{self._last_destination_name}' in memory to prevent immediate repeated navigation")

                        # Add location-specific speech feedback for successful navigation (only once)
                        if not self._arrival_announced:
                            if self._current_destination:
                                arrival_message = self._get_location_specific_arrival_message(self._current_destination)
                                self.tts_provider.add_pending_message(arrival_message)
                                logging.info(f"Generated location-specific arrival message for '{self._current_destination}'")
                            else:
                                self.tts_provider.add_pending_message(
                                    "Yaaay! I have reached my destination. Woof! Woof!"
                                )
                            self._arrival_announced = True  # Mark as announced to prevent repeats
                elif status_code in (5, 6):  # CANCELED or ABORTED
                    if self._nav_in_progress:
                        self._nav_in_progress = False
                        
                        # Clear goal tracking on failure/cancellation too
                        self._last_published_goal = None
                        self._last_destination_name = None
                        self._last_goal_time = 0.0
                        self._arrival_announced = False  # Reset arrival announcement
                        
                        # Do NOT re-enable AI mode on failure/cancellation
                        logging.warning(
                            "Navigation %s (code=%d) - AI mode remains disabled",
                            self.navigation_status,
                            status_code,
                        )
        else:
            logging.warning("Received empty navigation status message")

    def _publish_ai_status(self, enabled: bool):
        """
        Publish AI status to enable or disable AI mode during navigation.

        Parameters
        ----------
        enabled : bool
            True to enable AI mode, False to disable.
        """
        if self.ai_status_pub is None:
            logging.warning("AI status publisher not available")
            return

        try:
            header = prepare_header("map")
            status_msg = AIStatusRequest(
                header=header,
                request_id=String(str(uuid4())),
                code=1 if enabled else 0,
            )
            self.ai_status_pub.put(status_msg.serialize())
            logging.info(
                "AI mode %s during navigation", "enabled" if enabled else "disabled"
            )
        except Exception as e:
            logging.error(f"Error publishing AI status: {e}")

    def start(self):
        """
        Start the navigation provider by registering the message callback and starting the listener.
        """
        if self.session is None:
            logging.error(
                "Cannot start navigation provider; Zenoh session is not available."
            )
            return

        if not self.running:
            self.session.declare_subscriber(
                self.navigation_status_topic, self.navigation_status_message_callback
            )
            logging.info(
                "Subscribed to navigation status topic: %s",
                self.navigation_status_topic,
            )

            self.running = True
            logging.info("Navigation Provider started and listening for messages")
            return

        logging.warning("Navigation Provider is already running")

    def publish_goal_pose(
        self, pose: geometry_msgs.PoseStamped, destination_name: Optional[str] = None
    ):
        """
        Publish a goal pose to the navigation topic.
        Includes robust duplicate detection to prevent repeated goal publications.

        Parameters
        ----------
        pose : geometry_msgs.PoseStamped
            The goal pose to be published.
        destination_name : Optional[str]
            Name of the destination for speech feedback
        """
        if self.session is None:
            logging.error("Cannot publish goal pose; Zenoh session is not available.")
            return

        # Clear old destination tracking if enough time has passed
        self._clear_old_destination_if_needed()

        # Enhanced duplicate detection
        current_time = time.time()
        
        # Check if this is a duplicate goal by position and orientation
        if self._is_duplicate_goal(pose):
            logging.info(f"Duplicate goal detected for '{destination_name}' - skipping publication to prevent repeated navigation")
            return
            
        # Check if same destination name within reasonable time window (30 seconds)
        if (destination_name and 
            destination_name == self._last_destination_name and 
            hasattr(self, '_last_goal_time') and 
            current_time - self._last_goal_time < 30.0):
            logging.info(f"Same destination '{destination_name}' requested within 30 seconds - skipping to prevent repeated navigation")
            return
            
        # Check if navigation is already in progress to same destination
        if (self._nav_in_progress and 
            destination_name and 
            destination_name == self._last_destination_name):
            logging.info(f"Navigation to '{destination_name}' already in progress - skipping duplicate request")
            return

        # Track this goal to prevent future duplicates
        self._last_published_goal = pose
        self._last_destination_name = destination_name
        self._last_goal_time = current_time
        self._goal_publish_count += 1
        self._arrival_announced = False  # Reset arrival announcement for new goal
        
        # Store destination name for speech feedback
        self._current_destination = destination_name

        # Disable AI mode immediately when navigation goal is published
        if not self._nav_in_progress:
            self._publish_ai_status(enabled=False)
            logging.info("Navigation goal published - AI mode disabled immediately")

        self._nav_in_progress = True
        payload = ZBytes(pose.serialize())
        self.session.put(self.goal_pose_topic, payload)
        logging.info(f"Published goal pose #{self._goal_publish_count} for '{destination_name}' to topic: {self.goal_pose_topic}")

    def _is_duplicate_goal(self, new_pose: geometry_msgs.PoseStamped) -> bool:
        """
        Check if the new goal is a duplicate of the last published goal.
        
        Parameters
        ----------
        new_pose : geometry_msgs.PoseStamped
            The new pose to check for duplication
            
        Returns
        -------
        bool
            True if this is a duplicate goal, False otherwise
        """
        if self._last_published_goal is None:
            return False
            
        # Define tolerance for position and orientation comparison
        position_tolerance = 0.1  # 10 cm
        orientation_tolerance = 0.1  # ~5.7 degrees
        
        # Compare positions
        old_pos = self._last_published_goal.pose.position
        new_pos = new_pose.pose.position
        
        pos_diff = (
            (old_pos.x - new_pos.x) ** 2 +
            (old_pos.y - new_pos.y) ** 2 +
            (old_pos.z - new_pos.z) ** 2
        ) ** 0.5
        
        # Compare orientations (quaternions)
        old_ori = self._last_published_goal.pose.orientation
        new_ori = new_pose.pose.orientation
        
        ori_diff = (
            (old_ori.x - new_ori.x) ** 2 +
            (old_ori.y - new_ori.y) ** 2 +
            (old_ori.z - new_ori.z) ** 2 +
            (old_ori.w - new_ori.w) ** 2
        ) ** 0.5
        
        # Return True if both position and orientation are within tolerance
        return pos_diff < position_tolerance and ori_diff < orientation_tolerance

    def clear_goal_pose(self):
        """
        Clear/cancel all active navigation goals.
        Publishes to the cancel_goal topic to stop navigation.
        """
        if self.session is None:
            logging.error("Cannot cancel goal; Zenoh session is not available.")
            return

        try:
            # Send cancel request to Nav2
            # Empty payload should cancel all active goals
            cancel_payload = ZBytes(b"")
            self.session.put(self.cancel_goal_topic, cancel_payload)
            logging.info("Sent cancel all goals request to: %s", self.cancel_goal_topic)
            self._nav_in_progress = False
            
            # Clear goal tracking variables
            self._last_published_goal = None
            self._last_destination_name = None
            self._last_goal_time = 0.0
            self._arrival_announced = False  # Reset arrival announcement
            
        except Exception:
            logging.exception("Failed to cancel navigation goals")

    @property
    def navigation_state(self) -> str:
        """
        Get the current navigation state.

        Returns
        -------
        str
            The current navigation state as a string.
        """
        return self.navigation_status

    @property
    def is_navigating(self) -> bool:
        """
        Check if navigation is currently in progress.

        Returns
        -------
        bool
            True if navigation is in progress, False otherwise.
        """
        return self._nav_in_progress
    
    def _get_location_specific_arrival_message(self, destination: str) -> str:
        """
        Generate a location-specific arrival message with interesting context.
        
        Parameters
        ----------
        destination : str
            The destination name (e.g., 'kitchen', 'charger', 'front door')
            
        Returns
        -------
        str
            A contextual arrival message for the location
        """
        destination_lower = destination.lower()
        
        # Location-specific messages with interesting context
        location_messages = {
            'kitchen': [
                "Woof! I've reached the kitchen! This is where all the delicious smells come from. I can sense the aroma of past meals lingering in the air!",
                "Yaaay! I'm at the kitchen! This is the heart of the home where culinary magic happens. I bet there are tasty treats somewhere around here!",
                "I've arrived at the kitchen! This is where humans prepare their food. I love the sounds of cooking and the warmth that comes from this special place!"
            ],
            'charger': [
                "Perfect! I've reached my charging station! Time for some well-deserved rest and energy replenishment. This is my cozy power nap spot!",
                "Woof! I'm at the charger! This is my special recharging sanctuary where I restore my energy. Just like humans need sleep, I need my power station!",
                "I've made it to the charger! This is my energy oasis - the place where I get recharged and ready for more adventures with you!"
            ],
            'front door': [
                "I've reached the front door! This is the gateway between our cozy home and the exciting world outside. I can sense all the interesting smells from beyond!",
                "Woof! I'm at the front door! This is the portal where visitors come and go, bringing new adventures and stories into our home!",
                "I've arrived at the front door! This is the threshold of possibilities - the entrance to our safe haven and the exit to outdoor adventures!"
            ],
            'home': [
                "I'm back home! This is my favorite spot - the place where I feel safe, comfortable, and surrounded by everything familiar. There's no place like home!",
                "Woof! I've returned home! This is my sanctuary, my base of operations, where all my adventures begin and end. Home sweet home!",
                "I've reached home base! This is where I belong - surrounded by familiar sights, sounds, and that special feeling of being exactly where I should be!"
            ],
            'living room': [
                "I've made it to the living room! This is where relaxation happens - soft furniture, cozy corners, and the perfect spot for quality time together!",
                "Woof! I'm in the living room! This is the social heart of the home where everyone gathers to unwind and share their day!"
            ],
            'bedroom': [
                "I've reached the bedroom! This is the peaceful sanctuary where dreams are made and rest is found. Such a calm and serene space!",
                "Woof! I'm at the bedroom! This is the quiet retreat where humans recharge their energy, just like I do at my charging station!"
            ],
            'office': [
                "I've arrived at the office! This is the productivity zone where important work gets done. I can sense the focused energy of this space!",
                "Woof! I'm at the office! This is the thinking headquarters where ideas come to life and tasks get accomplished!"
            ],
            'garage': [
                "I've reached the garage! This is the mechanical sanctuary filled with tools, vehicles, and the scent of industry. What interesting machines live here!",
                "Woof! I'm in the garage! This is the workshop where things get fixed and projects come to life. I love the organized chaos of this space!"
            ],
            'bathroom': [
                "I've arrived at the bathroom! This is the refreshing sanctuary where humans get clean and tidy. The tiles feel cool under my sensors!",
                "Woof! I'm at the bathroom! This is the splashing zone where water flows and cleanliness happens. Such interesting echoes in here!"
            ],
            'dining room': [
                "I've reached the dining room! This is where families gather to share meals and stories. I can imagine all the wonderful conversations that happen here!",
                "Woof! I'm in the dining room! This is the feasting hall where delicious meals bring people together. The perfect spot for social bonding!"
            ],
            'laundry room': [
                "I've made it to the laundry room! This is the cleaning command center where fabrics get refreshed and renewed. I love the rhythmic sounds of the machines!",
                "Woof! I'm at the laundry room! This is where the magic of fresh, clean clothes happens. The warm, sudsy atmosphere is so cozy!"
            ],
            'balcony': [
                "I've reached the balcony! This is the outdoor sanctuary with fresh air and amazing views. What a perfect spot to observe the world beyond!",
                "Woof! I'm on the balcony! This is the sky-high perch where indoor comfort meets outdoor adventure. The breeze feels wonderful!"
            ],
            'garden': [
                "I've arrived in the garden! This is nature's playground filled with growing plants and fresh earth. I can sense all the life flourishing here!",
                "Woof! I'm in the garden! This is the green oasis where plants dance in the breeze and flowers share their fragrances. So peaceful!"
            ]
        }
        
        # Get location-specific messages or use default
        if destination_lower in location_messages:
            import random
            selected_message = random.choice(location_messages[destination_lower])
            logging.info(f"Selected location-specific message for '{destination}': {selected_message[:50]}...")
            return selected_message
        else:
            # Generic message for unknown locations
            generic_message = f"Yaaay! I have reached {destination}! This is an interesting place - I'm excited to explore and learn more about this location. Woof! Woof!"
            logging.info(f"Using generic arrival message for unknown location '{destination}'")
            return generic_message
    
    def _clear_old_destination_if_needed(self):
        """Clear destination name if enough time has passed since last navigation."""
        if (self._last_destination_name and 
            hasattr(self, '_last_goal_time') and 
            time.time() - self._last_goal_time > 60.0):  # Clear after 60 seconds
            logging.info(f"Clearing old destination '{self._last_destination_name}' after 60 seconds")
            self._last_destination_name = None
            self._last_goal_time = 0.0
    
    def get_navigation_debug_info(self) -> dict:
        """Get comprehensive navigation state information for debugging."""
        return {
            "nav_in_progress": self._nav_in_progress,
            "navigation_status": self.navigation_status,
            "last_destination_name": self._last_destination_name,
            "goal_publish_count": self._goal_publish_count,
            "has_last_published_goal": self._last_published_goal is not None,
            "last_goal_time": getattr(self, '_last_goal_time', 0.0),
            "current_time": time.time(),
            "running": self.running
        }
