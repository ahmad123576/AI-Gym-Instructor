# pose_detector.py
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Any

# Constants
DEFAULT_DETECTION_CONFIDENCE = 0.5
DEFAULT_TRACKING_CONFIDENCE = 0.5
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
LANDMARK_THICKNESS = 2
LANDMARK_RADIUS = 2
CONNECTION_THICKNESS = 2

class PoseDetector:
    """
    Pose detection using MediaPipe for real-time human pose estimation.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE):
        """
        Initialize the pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_pose(self, image: np.ndarray, draw: bool = False) -> Optional[Any]:
        """
        Detect pose landmarks in the input image and optionally draw them.
        
        Args:
            image: Input RGB image
            draw: Whether to draw landmarks on the image
            
        Returns:
            Pose landmarks object or None if no pose detected
        """
        results = self.pose.process(image)
        
        if draw and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=LANDMARK_COLOR, 
                    thickness=LANDMARK_THICKNESS, 
                    circle_radius=LANDMARK_RADIUS
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=CONNECTION_COLOR, 
                    thickness=CONNECTION_THICKNESS
                )
            )
        
        return results.pose_landmarks

    def __del__(self):
        """Clean up resources when object is destroyed."""
        if hasattr(self, 'pose'):
            self.pose.close()