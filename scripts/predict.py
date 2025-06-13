import tensorflow as tf
import cv2
import numpy as np
from utils.preprocess import preprocess_image, preprocess_image_advanced
from utils.classes import get_class_name
import os

model = None

def load_model_if_needed():
    global model
    if model is None:
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
        elif os.path.exists('model/model.h5'):
            model = tf.keras.models.load_model('model/model.h5')
        else:
            raise FileNotFoundError(
                "No se encontró ningún modelo entrenado. "
            )

def predict_image(img_path, use_advanced_preprocessing=True):
    try:
        load_model_if_needed()
        img = cv2.imread(img_path)
        if img is None:
            return "Error: No se pudo cargar la imagen", 0.0, []
        if use_advanced_preprocessing:
            img_processed = preprocess_image_advanced(img)
        else:
            img_processed = preprocess_image(img)
        pred = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)
        pred_probs = pred[0]

        top_3_indices = np.argsort(pred_probs)[-3:][::-1]
        top_3_predictions = [
            (get_class_name(idx), pred_probs[idx])
            for idx in top_3_indices
        ]

        best_class_id = int(np.argmax(pred_probs))
        best_confidence = float(np.max(pred_probs))
        best_class_name = get_class_name(best_class_id)
        
        return best_class_name, best_confidence, top_3_predictions
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0, []