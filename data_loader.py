import os, glob
import cv2
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

CAPTCHA_IMAGE_FOLDER = "captcha/captcha"
OUTPUT_FOLDER = "extracted_letter_images"

class Data_loader:
	
	def __init__(self, CAPTCHA_IMAGE_FOLDER, OUTPUT_FOLDER):
		self.CAPTCHA_IMAGE_FOLDER = CAPTCHA_IMAGE_FOLDER
		self.OUTPUT_FOLDER = OUTPUT_FOLDER
		
	def generate_data(self):
		captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
		counts = {}

		for (i, captcha_image_file) in enumerate(captcha_image_files):
			print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

			filename = os.path.basename(captcha_image_file)
			captcha_correct_text = os.path.splitext(filename)[0]
			image = cv2.imread(captcha_image_file)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
			thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contours = contours[0]

			letter_image_regions = []

			for contour in contours:

				(x, y, w, h) = cv2.boundingRect(contour)
				
				if w / h > 1.25:
					half_width = int(w / 2)
					letter_image_regions.append((x, y, half_width, h))
					letter_image_regions.append((x + half_width, y, half_width, h))
				else:
					letter_image_regions.append((x, y, w, h))

			if len(letter_image_regions) != 4:
				continue

			letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

			for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
				x, y, w, h = letter_bounding_box
				letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
				save_path = os.path.join(OUTPUT_FOLDER, letter_text)

				if not os.path.exists(save_path):
					os.makedirs(save_path)

				count = counts.get(letter_text, 1)
				p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
				cv2.imwrite(p, letter_image)
				
				counts[letter_text] = count + 1
				
	def split_data(self):
		data = []
		labels = []

		for image_file in paths.list_images(self.OUTPUT_FOLDER):
			
			image = cv2.imread(image_file)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			image = transform.resize(image, (20, 20))
			image = normalize(image)
			image = np.expand_dims(image, axis=2)

			label = image_file.split(os.path.sep)[-2]

			data.append(image)
			labels.append(label)
			
			data = np.array(data, dtype = "float32")
			labels = np.array(labels)

			lb = LabelBinarizer().fit(labels)
			labels = lb.transform(labels)
			
			with open(MODEL_LABELS_FILENAME, "rb") as f:
				lb = pickle.load(f)
			
			x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, shuffle = True)
			
			return x_train, y_train, x_test, y_test
		
	
		
	