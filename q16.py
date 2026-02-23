import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def get_data():
	image_size = (96, 96)
	batch_size = 16
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		"dataset/train",
		image_size=image_size,
		batch_size=batch_size,
		validation_split=0.2,
		subset="training",
		seed=122
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		"dataset/train",
		image_size=image_size,
		batch_size=batch_size,
		validation_split=0.2,
		subset="validation",
		seed=122
	)
	return train_ds, val_ds

def build_model():
	model = Sequential([
		Rescaling(1./255, input_shape=(96, 96, 3)),
		Conv2D(16, (3, 3), activation='relu'),
		MaxPooling2D((2, 2)),
		Flatten(),
		Dense(32, activation='relu'),
		Dense(10, activation='softmax')
	])
	model.compile(
		optimizer=Adam(learning_rate=0.001), 
		loss='sparse_categorical_crossentropy', 
		metrics=['accuracy']
	)
	return model

def main():
	train_ds, val_ds = get_data()
	train_ds, val_ds = train_ds.take(15), val_ds.take(5)
	
	model = build_model()
	
	print("\nTraining model and monitoring performance curves...")
	history = model.fit(train_ds, validation_data=val_ds, epochs=15)

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs_range = range(len(acc))

	plt.figure(figsize=(12, 5))

	# Accuracy Curve
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.legend(loc='lower right')

	# Loss Curve
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.legend(loc='upper right')

	plt.tight_layout()
	plt.savefig("performance_curves.png")
	plt.show()

if __name__ == "__main__":
	main()
