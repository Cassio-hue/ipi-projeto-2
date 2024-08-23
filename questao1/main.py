import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('brain.png', cv2.IMREAD_GRAYSCALE)

# Cortar a imagem
height, width = src.shape[:2]
image = src[:height-34, :width-35]

# Filtro passa baixa
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# Remove o ruído da imagem
median_filtered = cv2.medianBlur(blurred, 5)

# Binarização da imagem filtrada
threshold_value = 140
_, binary_image = cv2.threshold(median_filtered, threshold_value, 255, cv2.THRESH_BINARY)

# Remove excesso de pontos da imagem binarizada
kernel = np.ones((5, 5), np.uint8)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Utiliza a imagem fechada para destacar o tumor da imagem original
mask = closed_image == 255
color = (0, 255, 0)
imagem_colorida = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
imagem_colorida[mask] = color


plt.figure(100)
plt.subplot(1, 3, 1)
plt.imshow(median_filtered, cmap='gray')
plt.title('Imagem Filtrada')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imagem_colorida, cmap='gray')
plt.title('Tomor Detectado')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(closed_image, cmap='gray')
plt.title('Tomor Binarizado')
plt.axis('off')

plt.figure(200)
plt.hist(median_filtered.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histograma da Imagem Binarizada')
plt.xlabel('Intensidade do Pixel')
plt.ylabel('Frequência')
plt.grid(True)

plt.show()
