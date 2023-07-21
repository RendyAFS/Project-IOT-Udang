import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Definisi parameter untuk model CNN
batch_size = 32
epochs = 10
input_shape = (150, 150, 3)  # Ubah sesuai ukuran gambar Anda

# Definisi model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (udang atau bukan udang)
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Preprocessing dan Augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'D:\\BISA!\\Project-IOT-Udang\\data_latihan',  # Gunakan double backslash
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)


# Latih model
model.fit(train_generator, epochs=epochs)

# Simpan model
model.save('shrimp_detection_model.h5')

# Load model yang telah dilatih jika perlu
# model = tf.keras.models.load_model('shrimp_detection_model.h5')

# Sekarang, Anda dapat menggunakan model untuk mendeteksi bibit udang dalam gambar test.jpg
# Misalnya:
from PIL import Image

# Load gambar test
test_image = Image.open('test.jpg')
test_image = test_image.resize(input_shape[:2])  # Resize gambar sesuai input_shape model
test_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_array = test_array[np.newaxis, ...]  # Tambahkan dimensi batch

# Lakukan prediksi
prediction = model.predict(test_array)

# Jika nilai prediksi mendekati 1, maka itu merupakan bibit udang
if prediction[0][0] > 0.5:
    print("Bibit udang terdeteksi!")
else:
    print("Bukan bibit udang.")
