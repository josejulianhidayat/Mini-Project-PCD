import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def segment_image(image, k):
    # Ubah gambar menjadi array 2D dari piksel
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Implementasi clustering KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_values)
    
    # Ganti setiap nilai piksel dengan nilai pusat yang sesuai
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    segmented_image = centers[labels.flatten()]
    
    # Ubah kembali ke dimensi gambar asli
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

def main():
    # Memuat gambar
    image_path = 'bahan-gambar.jpg'
    image = load_image(image_path)

    # Jumlah cluster
    k = 3  # Anda bisa mengubah jumlah cluster

    # Segmentasi gambar
    segmented_image = segment_image(image, k)

    # Simpan dan tampilkan gambar yang telah tersegmentasi
    output_path = 'gambar_tersegmentasi.jpg'
    plt.imsave(output_path, segmented_image)
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
