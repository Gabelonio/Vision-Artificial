from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)

img = Image.open('increible.jfif')
imgGray = img.convert('L')
imgGray.save('imagen_gris.jpg')

arreglo_valores = []

img = cv2.imread("imagen_gris.jpg", 0) 
for i in range (img.shape[0]): 
    fila_valores = []
    for j in range (img.shape[1]): 
        fila_valores.append(img[i][j])
    arreglo_valores.append(fila_valores)

""" 
Imprimir imagen
imgGris = mpimg.imread('image.png') 
plt.imshow(imgGris, cmap = "gray")
plt.show()
 """

""" 
Obtener RGB individual
imgGris = Image.open('imagen_gris.jpg')
imagenRGB = imgGris.convert("RGB")
valorIndividual = imagenRGB.getpixel((500,150))
print(valorIndividual)
ancho, alto = imgGris.size  """


