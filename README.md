# Taller 3 IA
## Instrucciones de ejecución
* Primero se debe de ejecutar el archivo train_model.py (Advertencia, esta con 50 epocas, asi que mejor no correr, ya esta subido el archivo que genera este archivo, o cambiar la cantidad de epocas)
* segundo, verificar que se hayan generado los archivos model.h5, best_model.h5, training_history.npy
* ejecutar el archivo run_app.py para ver la interfaz y subir imagenes para que sean clasificadas
* una vez se sube la imagen no se clasifica automaticamente y da 3 posibilidades con un porcentage de probabilidad que sea esa opción
## Descripción del problema
Construir una red neuronal utilizando TensorFlow que sea capaz de clasificar imágenes de señales de tránsito. La solución deberá entrenarse con un conjunto de datos etiquetado y validarse correctamente para evaluar su rendimiento.
Se trabajará con el conjunto de datos German Traffic Sign Recognition Benchmark (GTSRB), el cual contiene miles de imágenes correspondientes a 43 clases distintas de señales de tránsito (alto, ceda el paso, límite de velocidad, entre otras).
La red deberá ser capaz de predecir correctamente la clase a la que pertenece una imagen de entrada nunca antes vista por el modelo. Como base, deberá de crear una interfaz gráfica sencilla que permita cargar imágenes de manera rápida y mostrar el resultado de la predicción.
## Arquitectura del modelo
* Optimizador: Adam (lr=0.001)
* Loss Function: Categorical Crossentropy
* Batch Size: 128
* Épocas: 50 (con Early Stopping)
* Validación: 20% del dataset
* Dropout: 0.25 en capas conv, 0.5 en capas densas
* ReduceLROnPlateau: Factor 0.5, paciencia 5 épocas
## Métricas obtenidas
* Época 1/50:  - loss: 2.1847 - accuracy: 0.3821 - val_accuracy: 0.7245
* Época 5/50:  - loss: 0.8234 - accuracy: 0.7456 - val_accuracy: 0.8567
* Época 15/50: - loss: 0.3412 - accuracy: 0.8934 - val_accuracy: 0.9201
* Época 32/50: - loss: 0.1876 - accuracy: 0.9387 - val_accuracy: 0.9423
* Mejor modelo: val_accuracy = 0.9423
## Conclusiones
Segun las pruebas que se hicieron, este modelo no detecta al 100% la imagen como debria de ser, falla bastante dependiendo del tipo de imagen que se suba, se intento con 5 imagenes y solo 2 de ellas detecto correctamente, en las otras, no era la que detectaba, pero la respuesta aparecia en el top 3 de la interfaz, pero con probabilidad baja de ser esa respuesta.
