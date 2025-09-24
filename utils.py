import cv2
import numpy as np
from typing import List, Tuple, Union

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (255, 0, 0)
TEXT_POSITION_CORRECT = (10, 30)
TEXT_POSITION_INCORRECT = (10, 60)

def calculate_angle(a: Union[List[float], np.ndarray], 
                   b: Union[List[float], np.ndarray], 
                   c: Union[List[float], np.ndarray]) -> float:
    """
    Calculate angle between three points (a, b, c) in degrees.
    
    Args:
        a: First point [x, y] or [x, y, z, visibility]
        b: Second point [x, y] or [x, y, z, visibility] 
        c: Third point [x, y] or [x, y, z, visibility]
        
    Returns:
        Angle in degrees between the three points
    """
    a = np.array(a[:2])  # Only use x, y coordinates
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ab = a - b
    bc = c - b
    
    # Avoid division by zero
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ab == 0 or norm_bc == 0:
        return 0.0
    
    dot_product = np.dot(ab, bc)
    cos_angle = dot_product / (norm_ab * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def draw_rep_count(image: np.ndarray, correct_reps: int, incorrect_reps: int) -> np.ndarray:
    """
    Draw correct and incorrect rep counts on the image.
    
    Args:
        image: Input image to draw on
        correct_reps: Number of correct repetitions
        incorrect_reps: Number of incorrect repetitions
        
    Returns:
        Image with rep counts drawn on it
    """
    # Draw correct reps in green
    correct_text = f"Correct: {correct_reps}"
    cv2.putText(
        image,
        correct_text,
        TEXT_POSITION_CORRECT,
        FONT,
        FONT_SCALE,
        GREEN_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    
    # Draw incorrect reps in red below
    incorrect_text = f"Incorrect: {incorrect_reps}"
    cv2.putText(
        image,
        incorrect_text,
        TEXT_POSITION_INCORRECT,
        FONT,
        FONT_SCALE,
        RED_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    return image
