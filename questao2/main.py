import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Ler a imagem "onion.png"
image = cv2.imread('onion.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Pre-processamento
# Aplicar um filtro de desfoque para suavizar a imagem
blurred_image = cv2.GaussianBlur(image_rgb, (7, 7), 0)

# Converter para o espaço de cores LAB para uma melhor segmentação
lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2LAB)

# 3. Aplicar o K-Means
pixel_values = lab_image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Critérios de parada do algoritmo K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 40 # Número de clusters (você pode ajustar esse valor)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertendo de volta os centros para valores de 8 bits
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(lab_image.shape)

# 4. Selecionar clusters que correspondem à cor da cebola
# Converter de volta para o espaço RGB para visualização
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

# Definir intervalos para a cor da cebola
# Valores podem precisar de ajustes
# upper_color = np.array([245, 221, 195])  
# lower_color = np.array([110, 90, 35])

# Segmentação da Pimenta
upper_color = np.array([172,59,61])  
lower_color = np.array([156,40,50])


# Criar uma máscara para a cor da cebola
mask = cv2.inRange(segmented_image_rgb, lower_color, upper_color)
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Exibir os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Imagem Original')
plt.imshow(image_rgb)

plt.subplot(1, 3, 2)
plt.title('Imagem Segmentada')
plt.imshow(segmented_image_rgb)

plt.subplot(1, 3, 3)
plt.title('Segmentação da Pimenta')
plt.imshow(result)

plt.show()
