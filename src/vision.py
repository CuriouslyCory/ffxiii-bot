import cv2
import numpy as np
import mss
from typing import Optional, Tuple, Dict, List

class FeatureMatcher:
    """
    Handles feature detection, description, and matching using ORB.
    Provides homography calculation for visual odometry.
    """
    def __init__(self):
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        # Initialize Brute-Force Matcher with Hamming distance (better for ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Detects keypoints and computes descriptors for the given image.
        """
        return self.orb.detectAndCompute(image, mask)

    def match_features(self, des1, des2, ratio_thresh=0.75) -> List[cv2.DMatch]:
        """
        Matches descriptors using KNN and applies Lowe's ratio test.
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
            
        matches = self.bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                
        return good_matches

    def compute_homography(self, kp1, kp2, good_matches, min_match_count=10):
        """
        Computes the Homography matrix if enough matches are found.
        Returns (Matrix, Mask)
        """
        if len(good_matches) < min_match_count:
            return None, None
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC to filter outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, mask

    def decompose_homography(self, M):
        """
        Decomposes the homography matrix to extract translation (dx, dy) and rotation (angle).
        Assumes the transformation is mostly 2D rotation and translation (affine-like).
        """
        if M is None:
            return 0.0, 0.0, 0.0
            
        # Extract translation
        dx = M[0, 2]
        dy = M[1, 2]
        
        # Extract rotation angle (approximate from top-left 2x2)
        # M = [[cos(a), -sin(a), dx], [sin(a), cos(a), dy], [0, 0, 1]]
        # atan2(sin(a), cos(a)) -> atan2(M[1,0], M[0,0])
        angle_rad = np.arctan2(M[1, 0], M[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        return dx, dy, angle_deg

class VisionEngine:
    """
    VisionEngine handles screen capture and template matching for game state detection.
    
    It supports defining Regions of Interest (ROIs) to optimize matching performance
    and provides fuzzy matching capabilities using OpenCV's matchTemplate.
    Now includes Feature Matching (ORB) for visual odometry.
    """
    
    def __init__(self, window_offset: Tuple[int, int] = (0, 0), resolution: Tuple[int, int] = (1920, 1080)):
        """
        Initializes the VisionEngine.
        
        :param window_offset: (x, y) offset of the game window on the screen.
        :param resolution: (width, height) of the game window.
        """
        self.window_offset = window_offset
        self.resolution = resolution
        self.sct = mss.mss()
        self.update_monitor()
        self.templates: Dict[str, np.ndarray] = {}
        
        # Initialize Feature Matcher
        self.feature_matcher = FeatureMatcher()

    def update_monitor(self):
        """Updates the monitor configuration for mss based on window_offset and resolution."""
        self.monitor = {
            "top": self.window_offset[1],
            "left": self.window_offset[0],
            "width": self.resolution[0],
            "height": self.resolution[1],
        }

    def capture_screen(self) -> np.ndarray:
        """
        Captures the current screen contents within the game window.
        
        :return: A numpy array representing the captured image in BGR format.
        """
        screenshot = self.sct.grab(self.monitor)
        # Convert to BGR format which OpenCV uses
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def load_template(self, name: str, path: str):
        """
        Loads a template image from disk.
        
        :param name: Unique identifier for the template.
        :param path: Path to the template image file.
        """
        template = cv2.imread(path)
        if template is None:
            raise FileNotFoundError(f"Template not found at {path}")
        self.templates[name] = template

    def find_template(
        self, 
        template_name: str, 
        image: np.ndarray, 
        threshold: float = 0.8,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Searches for a template within an image, optionally restricted to an ROI.
        
        :param template_name: Name of the pre-loaded template to search for.
        :param image: The source image to search in.
        :param threshold: Matching confidence threshold (0.0 to 1.0).
        :param roi: Optional (x, y, width, height) to restrict search.
        :return: (x, y, confidence) of the best match if above threshold, else None.
        """
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        search_image = image
        offset_x, offset_y = 0, 0
        
        if roi:
            x, y, w, h = roi
            # Ensure ROI is within image bounds
            y_end = min(y + h, image.shape[0])
            x_end = min(x + w, image.shape[1])
            search_image = image[y:y_end, x:x_end]
            offset_x, offset_y = x, y

        if search_image.shape[0] < template.shape[0] or search_image.shape[1] < template.shape[1]:
            return None

        result = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            return (max_loc[0] + offset_x, max_loc[1] + offset_y, max_val)
        
        return None

    def get_roi_slice(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Returns a slice of the image based on the ROI.
        
        :param image: Source image.
        :param roi: (x, y, width, height).
        :return: Sliced image.
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]

    def draw_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int], label: str = "", color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Draws an ROI rectangle on the image for debugging.
        
        :param image: Image to draw on.
        :param roi: (x, y, width, height).
        :param label: Text label for the ROI.
        :param color: BGR color of the rectangle.
        """
        x, y, w, h = roi
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_match(self, image: np.ndarray, match: Tuple[int, int, float], template_name: str, color: Tuple[int, int, int] = (0, 0, 255)):
        """
        Draws a match result on the image for debugging.
        
        :param image: Image to draw on.
        :param match: (x, y, confidence) from find_template.
        :param template_name: Name of the template that was matched.
        :param color: BGR color of the marker.
        """
        x, y, conf = match
        template = self.templates[template_name]
        h, w = template.shape[:2]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{template_name} ({conf:.2f})"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
