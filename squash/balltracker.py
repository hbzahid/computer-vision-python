from collections import deque

import numpy as np
import argparse
import cv2

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", help="path to the video file")
	ap.add_argument("-i", "--image", help="path to the image file")
	return vars(ap.parse_args())

def show_image(name, image):
    cv2.namedWindow(name)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def morph(image):
	kernel = np.ones((3, 3), dtype=np.uint8)
	image = cv2.dilate(image, kernel, iterations=2)
	image = cv2.erode(image, kernel, iterations=1)
	return image


class BallTracker:
	def __init__(self):
		self.backgroundSubtractor = cv2.createBackgroundSubtractorKNN(
			history=400, dist2Threshold=200, detectShadows=False
		)

		self._init_kalman_filter()
		self.measurement_history = deque(maxlen=10)
		self.prediction_history = deque(maxlen=10)

	def _init_kalman_filter(self):
		self.kalman = cv2.KalmanFilter(4, 2)
		self.kalman.measurementMatrix = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		], np.float32)
		self.kalman.transitionMatrix = np.array([
			[1, 0, 1, 0],
			[0, 1, 0, 1],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		], np.float32)
		self.processNoiseCov = 9e-2 * np.eye(4)
		self.measurementNoiseCov = 3e-5 * np.eye(2)

	def _draw_contrail(self, frame):
		for i in range(1, len(self.measurement_history)):
			thickness = int(np.sqrt(self.measurement_history.maxlen / float(i + 1)) * 1.5)
			cv2.line(frame, self.measurement_history[i - 1], self.measurement_history[i], (0, 0, 255), thickness)
		for i in range(1, len(self.prediction_history)):
			thickness = int(np.sqrt(self.prediction_history.maxlen / float(i + 1)) * 1.5)
			cv2.line(frame, self.prediction_history[i - 1], self.prediction_history[i], (0, 255, 0), thickness)

	def locate_ball_from_contours(self, contours):
		likely_balls = []
		for c in contours:
			(_, _), radius = cv2.minEnclosingCircle(c)
			x, y, w, h = cv2.boundingRect(c)
			cv2.putText(self.frame, 'w: %d, h: %d, radius: %d' % (w, h, radius), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
			if (abs(w - h) <= 10 and 5 <= radius <= 30):
				likely_balls.append(c)
		return likely_balls

	def detectBall(self, frame):
		"""
		Returns a contour representing the position of the ball in frame.
		If the ball isn't found or there are multiple candidates, returns None.
		"""

		foreground = self.backgroundSubtractor.apply(frame)
		# _, foreground = cv2.threshold(foreground, 20, 255, cv2.THRESH_BINARY)
		foreground = cv2.Canny(foreground, 100, 200)
		foreground = morph(foreground)
		show_image("foreground", foreground)
		contours, hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		print(len(contours))
		likely_balls = self.locate_ball_from_contours(contours)
		if len(likely_balls) == 1:
			return likely_balls[0]
		else:
			return None

	def track(self, videoFile):
		cap = cv2.VideoCapture(videoFile)
		if not cap.isOpened():
			raise Exception("Could not open video file {}".format(videoFile))

		measurement = np.zeros((2, 1), np.float32)
		prediction = np.zeros((2, 1), np.float32)
		
		detected_once = False
		ball_pos = None
		frameNum = -1
		while cap.isOpened():
			ret, frame = cap.read()
			self.frame = frame
			frameNum += 1

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
			
			ball_pos = self.detectBall(gray)
			if not detected_once:
				detected_once = ball_pos is not None

			if ball_pos is not None:
				(x, y, w, h) = bounding_box = cv2.boundingRect(ball_pos)
				measurement[0], measurement[1] = x, y
				self.kalman.correct(measurement)
				self.measurement_history.appendleft(tuple(measurement.astype('int').flatten()))
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			if detected_once:
				prediction = self.kalman.predict()
				self.prediction_history.appendleft(tuple(prediction.astype('int').flatten()[:2]))
			else:
				print("Trying to locate ball...")

			# self._draw_contrail(frame)

			# show_image("tracking", frame)


if __name__ == '__main__':
	args = parse_args()
	videoFile = args['video']

	tracker = BallTracker()
	tracker.track(videoFile)
