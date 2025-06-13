import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from scripts.predict import predict_image

def main():
    root = tk.Tk()
    root.title("Clasificador Avanzado de Señales de Tránsito")
    root.geometry("700x800")
    root.configure(bg='#f0f0f0')

    title_label = tk.Label(
        root, 
        text="🚦 Clasificador de Señales de Tránsito 🚦",
        font=("Helvetica", 18, "bold"),
        bg='#f0f0f0',
        fg='#2c3e50'
    )
    title_label.pack(pady=20)
    image_frame = tk.Frame(root, bg='#f0f0f0')
    image_frame.pack(pady=10)
    
    image_label = tk.Label(image_frame, bg='white', width=30, height=15)
    image_label.pack(padx=20, pady=20)

    options_frame = tk.Frame(root, bg='#f0f0f0')
    options_frame.pack(pady=10)

    prediction_method = tk.StringVar(value="Estándar")
    
    tk.Label(options_frame, text="Método de predicción:", bg='#f0f0f0', font=("Helvetica", 12)).pack()
    
    method_frame = tk.Frame(options_frame, bg='#f0f0f0')
    method_frame.pack(pady=5)
    
    tk.Radiobutton(
        method_frame, 
        text="Estándar", 
        variable=prediction_method, 
        value="standard",
        bg='#f0f0f0'
    ).pack(side=tk.LEFT, padx=10)

    result_frame = tk.Frame(root, bg='#f0f0f0')
    result_frame.pack(pady=20, padx=20, fill='x')
    
    tk.Label(result_frame, text="Predicción:", font=("Helvetica", 14, "bold"), bg='#f0f0f0').pack()
    
    main_result_label = tk.Label(
        result_frame, 
        text="Selecciona una imagen para clasificar",
        font=("Helvetica", 12),
        bg='white',
        relief='sunken',
        padx=10,
        pady=10,
        wraplength=600,
        justify='center'
    )
    main_result_label.pack(fill='x', pady=5)

    top3_frame = tk.LabelFrame(root, text="Top 3 Predicciones", font=("Helvetica", 12, "bold"), bg='#f0f0f0')
    top3_frame.pack(pady=10, padx=20, fill='x')
    
    top3_labels = []
    for i in range(3):
        label = tk.Label(
            top3_frame,
            text=f"{i+1}. ---",
            font=("Helvetica", 10),
            bg='#f0f0f0',
            anchor='w'
        )
        label.pack(fill='x', padx=10, pady=2)
        top3_labels.append(label)
    progress = ttk.Progressbar(root, mode='indeterminate')
    
    def cargar_imagen():
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de señal de tránsito",
            filetypes=[
                ("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp *.ppm"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                progress.pack(pady=10)
                progress.start()
                root.update()
                img = Image.open(file_path)
                
                img_display = img.copy()
                img_display.thumbnail((250, 250), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_display)
                image_label.configure(image=img_tk)
                image_label.image = img_tk
                
                method = prediction_method.get()

                nombre_clase, confianza, top_3 = predict_image(file_path, use_advanced_preprocessing=False)

                progress.stop()
                progress.pack_forget()

                if "Error" in nombre_clase:
                    main_result_label.configure(
                        text=nombre_clase,
                        bg='#ffebee',
                        fg='#c62828'
                    )
                    for i, label in enumerate(top3_labels):
                        label.configure(text=f"{i+1}. ---")
                else:
                    if confianza > 0.8:
                        bg_color = '#e8f5e8'
                        fg_color = '#2e7d32'
                    elif confianza > 0.5:
                        bg_color = '#fff3e0'
                        fg_color = '#ef6c00'
                    else:
                        bg_color = '#ffebee'
                        fg_color = '#c62828'
                    
                    main_result_label.configure(
                        text=f"🎯 {nombre_clase}\n📊 Confianza: {confianza:.1%}",
                        bg=bg_color,
                        fg=fg_color
                    )
                    for i, (clase, conf) in enumerate(top_3):
                        top3_labels[i].configure(
                            text=f"{i+1}. {clase} ({conf:.1%})"
                        )
                
            except Exception as e:
                progress.stop()
                progress.pack_forget()
                messagebox.showerror("Error", f"No se pudo procesar la imagen.\n{str(e)}")
                main_result_label.configure(
                    text=f"Error: {str(e)}",
                    bg='#ffebee',
                    fg='#c62828'
                )
    
    def probar_modelo():
        try:
            progress.pack(pady=10)
            progress.start()
            root.update()
            
            if test_model_with_sample():
                messagebox.showinfo("Éxito", "¡El modelo está funcionando correctamente!")
            else:
                messagebox.showerror("Error", "El modelo no está funcionando correctamente.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al probar el modelo: {str(e)}")
        finally:
            progress.stop()
            progress.pack_forget()
    
    # Botones
    buttons_frame = tk.Frame(root, bg='#f0f0f0')
    buttons_frame.pack(pady=20)
    
    load_button = tk.Button(
        buttons_frame,
        text="📁 Cargar Imagen",
        command=cargar_imagen,
        font=("Helvetica", 14, "bold"),
        bg='#3498db',
        fg='white',
        padx=20,
        pady=10,
        relief='raised'
    )
    load_button.pack(side=tk.LEFT, padx=10)
    
    test_button = tk.Button(
        buttons_frame,
        text="🔧 Probar Modelo",
        command=probar_modelo,
        font=("Helvetica", 12),
        bg='#95a5a6',
        fg='white',
        padx=15,
        pady=8
    )
    test_button.pack(side=tk.LEFT, padx=10)

    info_label = tk.Label(
        root,
        text="💡 Tip: Usa imágenes claras de señales de tránsito para mejores resultados\n"
             "El método 'Ensemble' combina múltiples técnicas para mayor precisión",
        font=("Helvetica", 9),
        bg='#f0f0f0',
        fg='#7f8c8d',
        justify='center'
    )
    info_label.pack(pady=(10, 20))
    
    root.mainloop()

if __name__ == "__main__":
    main()