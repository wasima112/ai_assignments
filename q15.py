import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_data():
	image_size = (96, 96)
	batch_size = 16
	
	# Load training and validation datasets
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
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model

def main():
	train_ds, val_ds = get_data()
	
	# Take smaller subset for faster CPU training
	train_ds, val_ds = train_ds.take(15), val_ds.take(5)
	
	model = build_model()

	# Define Callbacks
	early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
	checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)

	print("\nTraining with Callbacks...")
	history = model.fit(
		train_ds, 
		validation_data=val_ds, 
		epochs=20, 
		callbacks=[early_stop, checkpoint, reduce_lr]
	)

	# Plotting loss curves
	plt.figure(figsize=(10, 4))
	plt.plot(history.history['loss'], label='Train Loss')
	plt.plot(history.history['val_loss'], label='Val Loss')
	plt.title('Training Efficiency with Callbacks')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig("callback_performance.png")
	plt.show()

if __name__ == "__main__":
	main()
