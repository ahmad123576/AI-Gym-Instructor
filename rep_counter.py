import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from utils import calculate_angle, draw_rep_count

# Configure logging
logging.basicConfig(
    filename='rep_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
DEFAULT_FEEDBACK_DURATION = 60  # ~2 seconds at 30 fps
DEFAULT_ANGLE_BUFFER_SIZE = 3
DEFAULT_VISIBILITY_THRESHOLD = 0.5
DEFAULT_LOW_VISIBILITY_THRESHOLD = 0.15
DEFAULT_MIN_FRAMES_BETWEEN_REPS = 10
DEFAULT_ANGLE_SMOOTHING_BUFFER = 5

# Exercise-specific constants
SQUAT_KNEE_ANGLE_DOWN_CORRECT = 90
SQUAT_KNEE_ANGLE_UP = 160
SQUAT_BACK_ANGLE_MAX = 20
SQUAT_KNEE_AMPLITUDE_MIN_CORRECT = 70
SQUAT_KNEE_AMPLITUDE_MIN_COUNT = 20

BICEP_ELBOW_ANGLE_DOWN_CORRECT = 50
BICEP_ELBOW_ANGLE_UP = 160
BICEP_SHOULDER_ANGLE_MAX = 20
BICEP_ELBOW_AMPLITUDE_MIN_CORRECT = 110
BICEP_ELBOW_AMPLITUDE_MIN_COUNT = 40

OVERHEAD_ELBOW_ANGLE_DOWN = 100
OVERHEAD_ELBOW_ANGLE_UP_CORRECT = 170
OVERHEAD_ARM_ANGLE_MAX = 45
OVERHEAD_TORSO_ANGLE_MAX = 35
OVERHEAD_ELBOW_AMPLITUDE_MIN_CORRECT = 70
OVERHEAD_ELBOW_AMPLITUDE_MIN_COUNT = 20
OVERHEAD_HIP_Y_MAX_CHANGE = 100

class BaseRepCounter:
    """
    Base class for exercise rep counters with common functionality.
    """
    
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.is_correct = True
        self.rep_correct = True
        self.errors: List[str] = []
        self.feedback = ""
        self.feedback_frames = 0
        self.feedback_duration = DEFAULT_FEEDBACK_DURATION
        self.last_rep_snapshot: Optional[Dict[str, Any]] = None
        
    def _update_feedback(self, new_feedback: str) -> None:
        """Update feedback with persistence logic."""
        if new_feedback:
            self.feedback = new_feedback
            self.feedback_frames = self.feedback_duration
            logging.debug(f"{self.exercise_name} Feedback: {self.feedback}")
        elif self.feedback_frames > 0:
            self.feedback_frames -= 1
        else:
            self.feedback = ""
    
    def _check_visibility(self, landmarks: List[float], threshold: float = DEFAULT_VISIBILITY_THRESHOLD) -> bool:
        """Check if landmark visibility is above threshold."""
        return len(landmarks) > 3 and landmarks[3] > threshold
    
    def _smooth_angle(self, angle: float, buffer: List[float], buffer_size: int = DEFAULT_ANGLE_BUFFER_SIZE) -> float:
        """Smooth angle using a buffer to reduce noise."""
        buffer.append(angle)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        return sum(buffer) / len(buffer)

class SquatCounter(BaseRepCounter):
    """
    Counter for squat exercise with form analysis.
    """
    
    def __init__(self):
        super().__init__("Squat")
        self.state = "up"
        self.knee_angles: List[float] = []
        self.back_angles: List[float] = []
        self.knee_angle_down_correct = SQUAT_KNEE_ANGLE_DOWN_CORRECT
        self.knee_angle_up = SQUAT_KNEE_ANGLE_UP
        self.back_angle_max = SQUAT_BACK_ANGLE_MAX
        self.knee_amplitude_min_correct = SQUAT_KNEE_AMPLITUDE_MIN_CORRECT
        self.knee_amplitude_min_count = SQUAT_KNEE_AMPLITUDE_MIN_COUNT
        self.direction: Optional[str] = None
        self.prev_knee_angle: Optional[float] = None
        self.min_knee = float('inf')
        self.max_knee = float('-inf')
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        """
        Update squat counter with new frame data.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image: Current frame image
            
        Returns:
            Tuple of (processed_image, feedback_message)
        """
        new_feedback = ""
        
        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Squat: Missing landmarks")
            new_feedback = "Ensure full body is visible"
        else:
            h, w, _ = image.shape
            
            def get_point(idx: int) -> List[float]:
                """Extract landmark point with coordinates and visibility."""
                lm = landmarks.landmark[idx]
                return [lm.x * w, lm.y * h, lm.z * w, lm.visibility]

            # Extract key landmarks
            left_hip = get_point(23)
            left_knee = get_point(25)
            left_ankle = get_point(27)
            right_hip = get_point(24)
            right_knee = get_point(26)
            right_ankle = get_point(28)
            left_shoulder = get_point(11)

            # Calculate knee angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2

            # Smooth the knee angle
            smoothed_knee = self._smooth_angle(knee_angle, self.angle_buffer)

            back_vector = (left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1])
            vertical_vector = (0, -1)
            back_angle = calculate_angle(back_vector, (0, 0), vertical_vector)

            self.knee_angles.append(smoothed_knee)
            self.back_angles.append(back_angle)

            # Check form correctness
            frame_correct = (
                back_angle < self.back_angle_max and
                self._check_visibility(left_knee) and 
                self._check_visibility(right_knee)
            )
            
            if not frame_correct:
                if not self._check_visibility(left_knee) or not self._check_visibility(right_knee):
                    self.errors.append("Low knee visibility")
                    new_feedback = "Ensure knees are visible"
                elif back_angle >= self.back_angle_max:
                    self.errors.append(f"Back angle too large: {back_angle:.1f}°")
                    new_feedback = "Keep back straight"
                self.rep_correct = False

            self.is_correct = frame_correct

            if self.prev_knee_angle is None:
                self.prev_knee_angle = smoothed_knee
            else:
                current_dir = 'decreasing' if smoothed_knee < self.prev_knee_angle else 'increasing' if smoothed_knee > self.prev_knee_angle else None

                if self.direction == 'decreasing' and current_dir == 'increasing':
                    self.min_knee = min(self.min_knee, self.prev_knee_angle)
                    amplitude = self.max_knee - self.min_knee
                    if amplitude >= self.knee_amplitude_min_count:
                        if amplitude >= self.knee_amplitude_min_correct and self.min_knee <= self.knee_angle_down_correct and self.rep_correct:
                            self.correct_reps += 1
                            new_feedback = "Great job!"  # Appreciative feedback for correct rep
                        else:
                            self.incorrect_reps += 1
                            if self.min_knee > self.knee_angle_down_correct:
                                self.errors.append(f"Shallow squat: min knee {self.min_knee:.1f}° > {self.knee_angle_down_correct}°")
                                new_feedback = "Bend knees more"
                            elif amplitude < self.knee_amplitude_min_correct:
                                self.errors.append(f"Low knee amplitude: {amplitude:.1f}°")
                                new_feedback = "Squat deeper"
                        rep_status = "Correct" if self.rep_correct else "Wrong"
                        knee_avg = sum(self.knee_angles) / max(len(self.knee_angles), 1)
                        back_avg = sum(self.back_angles) / max(len(self.back_angles), 1)
                        # Snapshot metrics for external feedback
                        self.last_rep_snapshot = {
                            "exercise": "Squat",
                            "status": "correct" if self.rep_correct and amplitude >= self.knee_amplitude_min_correct and self.min_knee <= self.knee_angle_down_correct else "incorrect",
                            "knee_avg": knee_avg,
                            "knee_min": min(self.knee_angles or [0]),
                            "knee_max": max(self.knee_angles or [0]),
                            "back_avg": back_avg,
                            "targets": {
                                "knee_angle_down_correct": self.knee_angle_down_correct,
                                "knee_amplitude_min_correct": self.knee_amplitude_min_correct,
                                "back_angle_max": self.back_angle_max
                            },
                            "errors": list(self.errors)
                        }
                        log_message = (
                            f"Squat Rep {self.correct_reps + self.incorrect_reps}: {rep_status}, "
                            f"Avg Knee Angle: {knee_avg:.1f}°, Min: {min(self.knee_angles or [0]):.1f}°, Max: {max(self.knee_angles or [0]):.1f}°, "
                            f"Avg Back Angle: {back_avg:.1f}°, Correct Reps: {self.correct_reps}, Incorrect Reps: {self.incorrect_reps}"
                        )
                        if self.errors:
                            log_message += f", Errors: {', '.join(self.errors)}"
                        logging.info(log_message)
                        self.rep_correct = True
                        self.errors = []
                        self.knee_angles = []
                        self.back_angles = []
                        self.min_knee = float('inf')
                        self.max_knee = float('-inf')
                    self.direction = current_dir

                elif self.direction == 'increasing' and current_dir == 'decreasing':
                    self.max_knee = max(self.max_knee, self.prev_knee_angle)
                    self.direction = current_dir

                elif current_dir:
                    self.min_knee = min(self.min_knee, smoothed_knee)
                    self.max_knee = max(self.max_knee, smoothed_knee)
                    self.direction = current_dir

                self.prev_knee_angle = smoothed_knee

        # Update feedback using base class method
        self._update_feedback(new_feedback)

        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback

class BicepCurlCounter(BaseRepCounter):
    """
    Counter for bicep curl exercise with form analysis.
    """
    
    def __init__(self):
        super().__init__("Bicep Curl")
        self.state = "down"
        self.elbow_angles: List[float] = []
        self.shoulder_angles: List[float] = []
        self.elbow_angle_down_correct = BICEP_ELBOW_ANGLE_DOWN_CORRECT
        self.elbow_angle_up = BICEP_ELBOW_ANGLE_UP
        self.shoulder_angle_max = BICEP_SHOULDER_ANGLE_MAX
        self.elbow_amplitude_min_correct = BICEP_ELBOW_AMPLITUDE_MIN_CORRECT
        self.elbow_amplitude_min_count = BICEP_ELBOW_AMPLITUDE_MIN_COUNT
        self.min_frames_between_reps = DEFAULT_MIN_FRAMES_BETWEEN_REPS
        self.frames_since_last_rep = 0
        self.direction: Optional[str] = None
        self.prev_elbow_angle: Optional[float] = None
        self.min_elbow = float('inf')
        self.max_elbow = float('-inf')
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        """
        Update bicep curl counter with new frame data.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image: Current frame image
            
        Returns:
            Tuple of (processed_image, feedback_message)
        """
        new_feedback = ""
        
        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Bicep Curl: Missing landmarks")
            self.frames_since_last_rep += 1
            new_feedback = "Ensure arms are visible"
        else:
            h, w, _ = image.shape
            
            def get_point(idx: int) -> List[float]:
                """Extract landmark point with coordinates and visibility."""
                lm = landmarks.landmark[idx]
                return [lm.x * w, lm.y * h, lm.z * w, lm.visibility]

            # Extract key landmarks
            left_shoulder = get_point(11)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            right_shoulder = get_point(12)
            right_elbow = get_point(14)
            right_wrist = get_point(16)

            # Calculate elbow angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

            # Smooth the elbow angle with larger buffer for bicep curls
            smoothed_elbow = self._smooth_angle(elbow_angle, self.angle_buffer, DEFAULT_ANGLE_SMOOTHING_BUFFER)

            left_arm_vector = (left_shoulder[0] - left_elbow[0], left_shoulder[1] - left_elbow[1])
            vertical_vector = (0, -1)
            shoulder_angle = calculate_angle(left_arm_vector, (0, 0), vertical_vector)

            self.elbow_angles.append(smoothed_elbow)
            self.shoulder_angles.append(shoulder_angle)

            # Check form correctness
            frame_correct = (
                shoulder_angle < self.shoulder_angle_max and
                self._check_visibility(left_elbow) and 
                self._check_visibility(right_elbow)
            )
            
            if not frame_correct:
                if not self._check_visibility(left_elbow) or not self._check_visibility(right_elbow):
                    self.errors.append("Low elbow visibility")
                    new_feedback = "Ensure elbows are visible"
                elif shoulder_angle >= self.shoulder_angle_max:
                    self.errors.append(f"Shoulder angle too large: {shoulder_angle:.1f}°")
                    new_feedback = "Lower your shoulders"
                self.rep_correct = False

            self.is_correct = frame_correct

            if self.prev_elbow_angle is None:
                self.prev_elbow_angle = smoothed_elbow
                self.frames_since_last_rep += 1
            else:
                current_dir = 'decreasing' if smoothed_elbow < self.prev_elbow_angle else 'increasing' if smoothed_elbow > self.prev_elbow_angle else None
                logging.debug(f"Bicep Curl: Smoothed Elbow={smoothed_elbow:.1f}°, Prev={self.prev_elbow_angle:.1f}°, Direction={current_dir}, Frames Since Rep={self.frames_since_last_rep}")

                if self.direction == 'decreasing' and current_dir == 'increasing' and self.frames_since_last_rep >= self.min_frames_between_reps:
                    self.min_elbow = min(self.min_elbow, self.prev_elbow_angle)
                    amplitude = self.max_elbow - self.min_elbow
                    if amplitude >= self.elbow_amplitude_min_count:
                        if amplitude >= self.elbow_amplitude_min_correct and self.min_elbow <= self.elbow_angle_down_correct and self.rep_correct:
                            self.correct_reps += 1
                            new_feedback = "Great job!"
                        else:
                            self.incorrect_reps += 1
                            if self.min_elbow > self.elbow_angle_down_correct:
                                self.errors.append(f"Shallow curl: min elbow {self.min_elbow:.1f}° > {self.elbow_angle_down_correct}°")
                                new_feedback = "Bend elbow more"
                            elif amplitude < self.elbow_amplitude_min_correct:
                                self.errors.append(f"Low elbow amplitude: {amplitude:.1f}°")
                                new_feedback = "Curl higher"
                        rep_status = "Correct" if self.rep_correct else "Wrong"
                        elbow_avg = sum(self.elbow_angles) / max(len(self.elbow_angles), 1)
                        shoulder_avg = sum(self.shoulder_angles) / max(len(self.shoulder_angles), 1)
                        self.last_rep_snapshot = {
                            "exercise": "Bicep Curl",
                            "status": "correct" if self.rep_correct and amplitude >= self.elbow_amplitude_min_correct and self.min_elbow <= self.elbow_angle_down_correct else "incorrect",
                            "elbow_avg": elbow_avg,
                            "elbow_min": min(self.elbow_angles or [0]),
                            "elbow_max": max(self.elbow_angles or [0]),
                            "shoulder_avg": shoulder_avg,
                            "targets": {
                                "elbow_angle_down_correct": self.elbow_angle_down_correct,
                                "elbow_amplitude_min_correct": self.elbow_amplitude_min_correct,
                                "shoulder_angle_max": self.shoulder_angle_max
                            },
                            "errors": list(self.errors)
                        }
                        log_message = (
                            f"Bicep Curl Rep {self.correct_reps + self.incorrect_reps}: {rep_status}, "
                            f"Avg Elbow Angle: {elbow_avg:.1f}°, Min: {min(self.elbow_angles or [0]):.1f}°, Max: {max(self.elbow_angles or [0]):.1f}°, "
                            f"Avg Shoulder Angle: {shoulder_avg:.1f}°, Correct Reps: {self.correct_reps}, Incorrect Reps: {self.incorrect_reps}"
                        )
                        if self.errors:
                            log_message += f", Errors: {', '.join(self.errors)}"
                        logging.info(log_message)
                        self.rep_correct = True
                        self.errors = []
                        self.elbow_angles = []
                        self.shoulder_angles = []
                        self.min_elbow = float('inf')
                        self.max_elbow = float('-inf')
                        self.frames_since_last_rep = 0
                    self.direction = current_dir

                elif self.direction == 'increasing' and current_dir == 'decreasing':
                    self.max_elbow = max(self.max_elbow, self.prev_elbow_angle)
                    self.direction = current_dir

                elif current_dir:
                    self.min_elbow = min(self.min_elbow, smoothed_elbow)
                    self.max_elbow = max(self.max_elbow, smoothed_elbow)
                    self.direction = current_dir

                self.prev_elbow_angle = smoothed_elbow
                self.frames_since_last_rep += 1

        # Update feedback using base class method
        self._update_feedback(new_feedback)

        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback

class OverheadPressCounter(BaseRepCounter):
    """
    Counter for overhead press exercise with form analysis.
    """
    
    def __init__(self):
        super().__init__("Overhead Press")
        self.state = "down"
        self.elbow_angles: List[float] = []
        self.arm_angles: List[float] = []
        self.torso_angles: List[float] = []
        self.left_hip_ys: List[float] = []
        self.right_hip_ys: List[float] = []
        self.elbow_angle_down = OVERHEAD_ELBOW_ANGLE_DOWN
        self.elbow_angle_up_correct = OVERHEAD_ELBOW_ANGLE_UP_CORRECT
        self.arm_angle_max = OVERHEAD_ARM_ANGLE_MAX
        self.torso_angle_max = OVERHEAD_TORSO_ANGLE_MAX
        self.elbow_amplitude_min_correct = OVERHEAD_ELBOW_AMPLITUDE_MIN_CORRECT
        self.elbow_amplitude_min_count = OVERHEAD_ELBOW_AMPLITUDE_MIN_COUNT
        self.hip_y_max_change = OVERHEAD_HIP_Y_MAX_CHANGE
        self.direction: Optional[str] = None
        self.prev_elbow_angle: Optional[float] = None
        self.min_elbow = float('inf')
        self.max_elbow = float('-inf')
        self.angle_buffer: List[float] = []

    def update(self, landmarks, image) -> Tuple[np.ndarray, str]:
        """
        Update overhead press counter with new frame data.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image: Current frame image
            
        Returns:
            Tuple of (processed_image, feedback_message)
        """
        new_feedback = ""
        
        if not landmarks:
            self.is_correct = False
            self.errors.append("Missing landmarks")
            logging.debug("Overhead Press: Missing landmarks")
            new_feedback = "Ensure full body is visible"
        else:
            h, w, _ = image.shape
            
            def get_point(idx: int) -> List[float]:
                """Extract landmark point with coordinates and visibility."""
                lm = landmarks.landmark[idx]
                return [lm.x * w, lm.y * h, lm.z * w, lm.visibility]

            # Extract key landmarks
            left_shoulder = get_point(11)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            right_shoulder = get_point(12)
            right_elbow = get_point(14)
            right_wrist = get_point(16)
            left_hip = get_point(23)
            right_hip = get_point(24)

            # Calculate elbow angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

            # Smooth the elbow angle
            smoothed_elbow = self._smooth_angle(elbow_angle, self.angle_buffer)

            left_arm_vector = (left_wrist[0] - left_shoulder[0], left_wrist[1] - left_shoulder[1])
            right_arm_vector = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
            vertical_vector = (0, -1)
            left_arm_angle = calculate_angle(left_arm_vector, (0, 0), vertical_vector)
            right_arm_angle = calculate_angle(right_arm_vector, (0, 0), vertical_vector)
            arm_angle = (left_arm_angle + right_arm_angle) / 2

            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            torso_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
            torso_angle = calculate_angle(torso_vector, (0, 0), vertical_vector)

            self.elbow_angles.append(smoothed_elbow)
            self.arm_angles.append(arm_angle)
            self.torso_angles.append(torso_angle)
            self.left_hip_ys.append(left_hip[1])
            self.right_hip_ys.append(right_hip[1])

            # Check form correctness
            frame_correct = (
                arm_angle < self.arm_angle_max and
                torso_angle < self.torso_angle_max and
                self._check_visibility(left_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD) and 
                self._check_visibility(right_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD)
            )
            
            if not frame_correct:
                if not self._check_visibility(left_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD) or not self._check_visibility(right_wrist, DEFAULT_LOW_VISIBILITY_THRESHOLD):
                    self.errors.append(f"Low wrist visibility: L={left_wrist[3]:.2f}, R={right_wrist[3]:.2f}")
                    new_feedback = "Ensure wrists are visible"
                elif torso_angle >= self.torso_angle_max:
                    self.errors.append(f"Torso angle too large: {torso_angle:.1f}°")
                    new_feedback = "Keep torso upright"
                elif arm_angle >= self.arm_angle_max:
                    self.errors.append(f"Arm angle too large: {arm_angle:.1f}°")
                    new_feedback = "Align arms vertically"
                self.rep_correct = False

            self.is_correct = frame_correct

            if self.prev_elbow_angle is None:
                self.prev_elbow_angle = smoothed_elbow
            else:
                current_dir = 'increasing' if smoothed_elbow > self.prev_elbow_angle else 'decreasing' if smoothed_elbow < self.prev_elbow_angle else None

                if self.direction == 'increasing' and current_dir == 'decreasing':
                    self.max_elbow = max(self.max_elbow, self.prev_elbow_angle)
                    amplitude = self.max_elbow - self.min_elbow
                    if amplitude >= self.elbow_amplitude_min_count:
                        hip_y_change = max(self.left_hip_ys + self.right_hip_ys or [0]) - min(self.left_hip_ys + self.right_hip_ys or [0])
                        if amplitude >= self.elbow_amplitude_min_correct and self.max_elbow >= self.elbow_angle_up_correct and hip_y_change <= self.hip_y_max_change and self.rep_correct:
                            self.correct_reps += 1
                            new_feedback = "Great job!"
                        else:
                            self.incorrect_reps += 1
                            if hip_y_change > self.hip_y_max_change:
                                self.errors.append(f"High hip movement: {hip_y_change:.1f} pixels")
                                new_feedback = "Keep hips stable"
                            elif self.max_elbow < self.elbow_angle_up_correct:
                                self.errors.append(f"Short press: max elbow {self.max_elbow:.1f}° < {self.elbow_angle_up_correct}°")
                                new_feedback = "Raise arms higher"
                            elif amplitude < self.elbow_amplitude_min_correct:
                                self.errors.append(f"Low elbow amplitude: {amplitude:.1f}°")
                                new_feedback = "Extend arms fully"
                        rep_status = "Correct" if self.rep_correct else "Wrong"
                        elbow_avg = sum(self.elbow_angles) / max(len(self.elbow_angles), 1)
                        arm_avg = sum(self.arm_angles) / max(len(self.arm_angles), 1)
                        torso_avg = sum(self.torso_angles) / max(len(self.torso_angles), 1)
                        self.last_rep_snapshot = {
                            "exercise": "Overhead Press",
                            "status": "correct" if self.rep_correct and amplitude >= self.elbow_amplitude_min_correct and self.max_elbow >= self.elbow_angle_up_correct and hip_y_change <= self.hip_y_max_change else "incorrect",
                            "elbow_avg": elbow_avg,
                            "elbow_min": min(self.elbow_angles or [0]),
                            "elbow_max": max(self.elbow_angles or [0]),
                            "arm_avg": arm_avg,
                            "torso_avg": torso_avg,
                            "hip_y_change": hip_y_change,
                            "targets": {
                                "elbow_angle_up_correct": self.elbow_angle_up_correct,
                                "elbow_amplitude_min_correct": self.elbow_amplitude_min_correct,
                                "arm_angle_max": self.arm_angle_max,
                                "torso_angle_max": self.torso_angle_max,
                                "hip_y_max_change": self.hip_y_max_change
                            },
                            "errors": list(self.errors)
                        }
                        log_message = (
                            f"Overhead Press Rep {self.correct_reps + self.incorrect_reps}: {rep_status}, "
                            f"Avg Elbow Angle: {elbow_avg:.1f}°, Min: {min(self.elbow_angles or [0]):.1f}°, Max: {max(self.elbow_angles or [0]):.1f}°, "
                            f"Avg Arm Angle: {arm_avg:.1f}°, Avg Torso Angle: {torso_avg:.1f}°, Hip Y Change: {hip_y_change:.1f} pixels, "
                            f"Correct Reps: {self.correct_reps}, Incorrect Reps: {self.incorrect_reps}"
                        )
                        if self.errors:
                            log_message += f", Errors: {', '.join(self.errors)}"
                        logging.info(log_message)
                        self.rep_correct = True
                        self.errors = []
                        self.elbow_angles = []
                        self.arm_angles = []
                        self.torso_angles = []
                        self.left_hip_ys = []
                        self.right_hip_ys = []
                        self.min_elbow = float('inf')
                        self.max_elbow = float('-inf')
                    self.direction = current_dir

                elif self.direction == 'decreasing' and current_dir == 'increasing':
                    self.min_elbow = min(self.min_elbow, self.prev_elbow_angle)
                    self.direction = current_dir

                elif current_dir:
                    self.min_elbow = min(self.min_elbow, smoothed_elbow)
                    self.max_elbow = max(self.max_elbow, smoothed_elbow)
                    self.direction = current_dir

                self.prev_elbow_angle = smoothed_elbow

        # Update feedback using base class method
        self._update_feedback(new_feedback)

        return draw_rep_count(image, self.correct_reps, self.incorrect_reps), self.feedback