import cv2
import imutils


def resize_to_fit(self, image, width, height):
	"""
	A helper function to resize an image to fit within a given size
	:param image: image to resize
	:param width: desired width in pixels
	:param height: desired height in pixels
	:return: the resized image
	"""
	(h, w) = image.shape[:2]

	if w > h:
		image = imutils.resize(image, width=width)
	else:
		image = imutils.resize(image, height=height)

	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	return image