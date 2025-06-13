import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.preprocess import preprocess_image
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = 'data/GTSRB/Final_Training/Images/'

def load_data():
    images = []
    labels = []
    
    for i in range(43):
        path = os.path.join(DATASET_PATH, format(i, '05d'))
        if not os.path.exists(path):
            print(f"Advertencia: La carpeta {path} no existe")
            continue
            
        class_images = 0
        for img_file in os.listdir(path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    processed_img = preprocess_image(img)
                    images.append(processed_img)
                    labels.append(i)
                    class_images += 1
    return np.array(images), np.array(labels)

def create_improved_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(43, activation='softmax')
    ])
    
    return model

def train_model():
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: No se cargaron imágenes. Verifica la ruta del dataset.")
        return
    
    print(f"Datos cargados: {X.shape}")

    y_cat = to_categorical(y, 43)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )
    
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Validación: {X_val.shape}")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False, 
        fill_mode='nearest'
    )
    model = create_improved_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("Iniciando entrenamiento...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        steps_per_epoch=len(X_train) // 128,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    model.save('model/model.h5')
    print("Modelo guardado como 'model/model.h5'")

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Precisión en validación: {val_accuracy:.4f}")

    np.save('model/training_history.npy', history.history)
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()