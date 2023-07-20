import os
import numpy as np
import tensorflow as tf

def save_model(model, save_dir):
    # Buat direktori jika belum ada
    os.makedirs(save_dir, exist_ok=True)

    # Simpan model sebagai format SavedModel
    tf.saved_model.save(model, save_dir)

if __name__ == "__main__":
    # Contoh data latih dan label
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)

    # Contoh pembuatan model dan training (silakan sesuaikan dengan model Anda)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10)

    # Ganti 'D:/BISA!/OpenCvUdang/model/' dengan jalur tempat Anda ingin menyimpan model
    model_save_path = 'D:/BISA!/OpenCvUdang/model/'
    save_model(model, model_save_path)
