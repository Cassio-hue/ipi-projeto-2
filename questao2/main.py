import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
src = cv2.imread('onion.png', cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# Converter a imagem para uma matriz de duas dimensões
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# Definir os critérios de parada e o número de clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
num_clusters = 3

# Aplicar o k-means
_, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converter os centros para valores de 8 bits e as labels para imagem
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Identificar o cluster mais vermelho
red_cluster_index = np.argmax(centers[:, 0] - centers[:, 1] - centers[:, 2])

# Criar uma máscara para o cluster vermelho
mask = (labels.flatten() == red_cluster_index)
mask = mask.reshape(image.shape[:2])

# Aplicar a máscara para segmentar a pimenta vermelha
segmented_red_pepper = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

# Criar um layout com duas colunas para exibir as imagens lado a lado
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_red_pepper)
plt.title('Pimenta Vermelha Segmentada')
plt.axis('off')

plt.show()
