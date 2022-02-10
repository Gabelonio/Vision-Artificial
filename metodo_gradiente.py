import math
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
    opt = int(input("1. Imagen animada\n2. Imagen del grupo\nOpción: "))

    if opt == 1:
        img = cv2.imread("mrincreible.jpeg")  # Lea la imagen aquí
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.array ([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    elif opt == 2:
        img = cv2.imread("grupoimg.jpeg")  # Lea la imagen aquí
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.array ([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    else:
        print("Opción no valida")
        quit()

    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.figure(1)
    plt.subplot(121),plt.imshow(img2),plt.title("Original")
    plt.subplot(122),plt.imshow(gray, cmap='gray'),plt.title("Grises")

    filterHorizontal = np.array ([[-1, 0, 1]]) # Este es el filtro establecido, que es el núcleo de convolución
    # filterHorizontal = np.array ([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 5, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]]) # Este es el filtro establecido, que es el núcleo de convolución

    horizontalBorders = matrixToCovolve(gray, filterHorizontal)

    if opt == 1:
        img = cv2.imread("mrincreible.jpeg")  # Lea la imagen aquí
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.array ([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    elif opt == 2:
        img = cv2.imread("grupoimg.jpeg")  # Lea la imagen aquí
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.array ([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    else:
        print("Opción no valida")
        quit()

    filterVertical = np.array([[-1], [0], [1]])

    verticalBorders = matrixToCovolve(gray, filterVertical)

    wholeBorders = processWholeBorders(horizontalBorders, verticalBorders)

    # cv2.imshow('Bordes Horizontales', horizontalBorders)
    # cv2.imshow('Bordes Verticales', verticalBorders)
    # cv2.imshow('Suma de Bordes', wholeBorders)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.figure(2)
    plt.subplot(131),plt.imshow(horizontalBorders, cmap='gray'),plt.title("Bordes Horizontales")
    plt.subplot(132),plt.imshow(verticalBorders, cmap='gray'),plt.title("Bordes Verticales")
    plt.subplot(133),plt.imshow(wholeBorders, cmap='gray'),plt.title("Suma de Bordes")

    plt.show()


def matrixToCovolve(matrix, kernel):
    yMatrix, xMatrix = np.shape(matrix)
    yKernel, xKernel = np.shape(kernel)
    additionalRows = math.floor(yKernel/2)
    additionalColumns = math.floor(xKernel/2)

    copyMatrix = np.zeros((yMatrix + 2*additionalRows, xMatrix + 2*additionalColumns))

    for i in range(additionalRows, yMatrix + additionalRows):
        for j in range(additionalColumns, xMatrix + additionalColumns):
            copyMatrix[i][j] = copyMatrix[i][j] + \
                matrix[i - additionalRows][j - additionalColumns]

    for i in range(additionalRows, yMatrix + additionalRows):
        for j in range(additionalColumns, xMatrix + additionalColumns):
            tempMatrix = np.zeros(np.shape(kernel))
            ytempMatrix, xtempMatrix = np.shape(tempMatrix)
            if additionalRows > 0:
                for p in range(-additionalRows, 0, 1):
                    if additionalColumns > 0:
                        for q in range(-additionalColumns, 0, 1):
                            ytempMatrix = ytempMatrix - 1
                            xtempMatrix = xtempMatrix - 1
                            tempMatrix[ytempMatrix][xtempMatrix] = copyMatrix[i - p][j - q]
                    else:
                        ytempMatrix = ytempMatrix - 1
                        tempMatrix[ytempMatrix][0] = copyMatrix[i - p][j]
            elif additionalColumns > 0:
                    for q in range(-additionalColumns, 0, 1):
                        xtempMatrix = xtempMatrix - 1
                        tempMatrix[0][xtempMatrix] = copyMatrix[i][j - q]
            else:
                print("Ambos parámetros son cero")

            ytempMatrix = ytempMatrix - 1
            xtempMatrix = xtempMatrix - 1
            tempMatrix[ytempMatrix][xtempMatrix] = copyMatrix[i][j]

            if additionalRows > 0:
                for p in range(-additionalRows, 0, 1):
                    if additionalColumns > 0:
                        for q in range(-additionalColumns, 0, 1):
                            ytempMatrix = ytempMatrix - 1
                            xtempMatrix = xtempMatrix - 1
                            tempMatrix[ytempMatrix][xtempMatrix] = copyMatrix[i + p][j + q]
                    else:
                        ytempMatrix = ytempMatrix - 1
                        tempMatrix[ytempMatrix][0] = copyMatrix[i + p][j]
            elif additionalColumns > 0:
                    for q in range(-additionalColumns, 0, 1):
                        xtempMatrix = xtempMatrix - 1
                        tempMatrix[0][xtempMatrix] = copyMatrix[i][j + q]
            else:
                print("Ambos parámetros son cero")

            result = covolveOne(tempMatrix, kernel)

            copyOfCopyMatrix = matrix

            copyOfCopyMatrix[i - additionalRows][j - additionalColumns] = result

    return copyOfCopyMatrix

def covolveOne(tempMatrix, kernel):
    ykernel, xkernel = np.shape(kernel)
    result = np.ones(np.shape(kernel))
    acum = 0

    for i in range(ykernel):
        for j in range(xkernel):
            result[i][j] = tempMatrix[i][j]*kernel[i][j]

    for i in range(ykernel):
        for j in range(xkernel):
            acum = acum + result[i][j]

    if acum < 0:
        return 0
    elif acum > 255:
        return 255
    else:
        return acum

def processWholeBorders(horizontalBorders, verticalBorders):
    yhorizontalBorders, xhorizontalBorders = np.shape(horizontalBorders)
    result = np.ones(np.shape(horizontalBorders))
    umbral = int(input("Inserte el umbral para la binarización: "))

    for i in range(yhorizontalBorders):
        for j in range(xhorizontalBorders):
            result[i][j] = math.sqrt(math.pow(horizontalBorders[i][j], 2) + math.pow(verticalBorders[i][j], 2))

            if result[i][j] < umbral:
                result[i][j] = 0
            elif result[i][j] >= umbral:
                result[i][j] = 255

    return result

if __name__ == '__main__':
    main()
