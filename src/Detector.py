import mediapipe as mp
mp_hands = mp.solutions.hands

class Detector:
    def __init__(self, num_frames_to_track=5, movement_threshold=10):
        self.num_frames_to_track = num_frames_to_track
        self.movement_threshold = movement_threshold
        self.prev_frames = []

    def track_frame(self, landmarks):
        hand_center_x = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x + landmarks[mp_hands.HandLandmark.PINKY_TIP].x) / 2

        movement_detected = False

        if len(self.prev_frames) == self.num_frames_to_track:
            prev_avg_center = sum(self.prev_frames) / len(self.prev_frames)
            movement = abs(hand_center_x - prev_avg_center)

            if movement > self.movement_threshold:
                movement_detected = True

            self.prev_frames.pop(0)

        self.prev_frames.append(hand_center_x)

        return movement_detected
