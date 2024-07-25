import os
import random

# Directorio que contiene las imágenes
directorio = "E:\\UPM\\TFG_SOFTWARE\\Proyecto\\Parte2\\images"

# Obtener la lista de archivos en el directorio
lista_archivos = os.listdir(directorio)

# Seleccionar aleatoriamente 24000 imágenes
imagenes_seleccionadas = random.sample(lista_archivos, 24000)

# Eliminar las imágenes que no están seleccionadas
for archivo in lista_archivos:
    if archivo not in imagenes_seleccionadas:
        os.remove(os.path.join(directorio, archivo))

print("Se han eliminado todas las imágenes que no son las 24000 seleccionadas.")
