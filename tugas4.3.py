import imageio.v2 as img
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def hist_equal(image):
    # Hitung histogram dan bins
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    # Hitung cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalisasi CDF
    cdf_normal = (cdf / cdf.max()) * 255
    # Terapkan ekualisasi histogram
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normal)
    return image_equalized.reshape(image.shape).astype(np.uint8)

# Baca gambar
image = img.imread("C:\\Users\\muham\\Downloads\\nature.jpg")
print("Gambar berhasil dimuat!")

# Ekualisasi histogram
result = hist_equal(image)
print("Ekualisasi histogram berhasil diterapkan!")

# Terapkan filter Gaussian ke gambar yang sudah diekualisasi
result_smoothed = ndimage.gaussian_filter(result, sigma=1)
print("Filter Gaussian (scipy ndimage) berhasil diterapkan!")

# Hitung histogram
hist_img, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
hist_result, bins = np.histogram(result.flatten(), bins=256, range=[0, 256])
hist_result_smoothed, bins = np.histogram(result_smoothed.flatten(), bins=256, range=[0, 256])
print("Histogram berhasil dihitung!")

# Plot hasil
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')

plt.subplot(2, 3, 2)
plt.imshow(result, cmap='gray')
plt.title('Gambar Setelah Ekualisasi')

plt.subplot(2, 3, 3)
plt.imshow(result_smoothed, cmap='gray')
plt.title('Gambar Ekualisasi Setelah Filter Gaussian')

plt.subplot(2, 3, 4)
plt.plot(hist_img)
plt.title('Histogram Gambar Asli')

plt.subplot(2, 3, 5)
plt.plot(hist_result)
plt.title('Histogram Gambar Setelah Ekualisasi')

plt.subplot(2, 3, 6)
plt.plot(hist_result_smoothed)
plt.title('Histogram Gambar Setelah Ekualisasi dan Filter Gaussian')

plt.show()

print("Kode berhasil dijalankan!")
