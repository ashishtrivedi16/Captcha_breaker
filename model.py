import cv2
import pickle
import imutils
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils import resize_to_fit
import matplotlib.pyplot as plt


class Model:
	
	def __init__(self, lr, batch_size, epochs, img_width, img_height, MODEL_FILENAME, MODEL_LABELS_FILENAME):
		self.lr = lr
		self.batch_size = batch_size
		self.epochs = epochs
		self.img_width = img_width
		self.img_height = img_height
		self.MODEL_FILENAME = MODEL_FILENAME
		self.MODEL_LABELS_FILENAME = MODEL_LABELS_FILENAME
		
	def model(self):
		
		model = Sequential()
		model.add(Conv2D(16, 3, input_shape = (self.img_width, self.img_height, 1), activation = 'relu'))
		model.add(MaxPool2D(pool_size = (2, 2)))
		model.add(Conv2D(64, 3, activation = 'relu'))
		model.add(MaxPool2D(pool_size = (2, 2)))
		model.add(Flatten())
		model.add(Dense(256, activation = 'relu'))
		model.add(Dropout(0.20))
		model.add(Dense(32, activation = "softmax"))

		optimizer = RMSprop(lr = self.lr)
		
		model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
		
		return model
	
	def train_model(self, model, x_train, y_train, x_test, y_test):
		reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                              patience = 2, min_lr = 0.000001)
		early_stop = EarlyStopping(monitor = "val_loss", patience = 5)
		
		history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = self.epochs, batch_size = self.batch_size, callbacks = [reduce_lr, early_stop])
		
		model.save(self.MODEL_FILENAME)
		return history
	
	def plot_history(self, history):
		
		plt.subplot(1, 2, 1)
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		
		plt.subplot(1, 2, 2)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		
	def load_model(self):
		model = load_model(self.MODEL_FILENAME)
		return model
		
		
	def predict_model(self, model, image_list):
		for image_file in image_list:
			image = cv2.imread(image_file)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
			thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
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

			letter_image_regions = sorted(letter_image_regions, key = lambda x: x[0])

			output = cv2.merge([image] * 3)
			predictions = []

			for letter_bounding_box in letter_image_regions:
				x, y, w, h = letter_bounding_box

				letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
				letter_image = resize_to_fit(letter_image, 20, 20)
				letter_image = np.expand_dims(letter_image, axis=2)
				letter_image = np.expand_dims(letter_image, axis=0)

				prediction = model.predict(letter_image)

				with open(MODEL_LABELS_FILENAME, "rb") as f:
					lb = pickle.load(f)
				
				letter = lb.inverse_transform(prediction)[0]
				predictions.append(letter)

				cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
				cv2.putText(output, prediction, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 225), 1)

			captcha_text = "".join(predictions)
			print("CAPTCHA text is: {}".format(captcha_text))

			plt.imshow(output, 'gray')
			plt.show()

