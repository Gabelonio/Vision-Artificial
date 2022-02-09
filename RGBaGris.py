from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

img = Image.open('imagen.jpg')
imgGray = img.convert('L')
imgGray.save('imagen_gris.jpg')

imgGris = Image.open('imagen_gris.jpg')
imagenRGB = imgGris.convert("RGB")
valorIndividual = imagenRGB.getpixel((500,150))
print(valorIndividual)

""" 
Imprimir imagen
imgGris = mpimg.imread('image.png') 
plt.imshow(imgGris, cmap = "gray")
plt.show()
 """


