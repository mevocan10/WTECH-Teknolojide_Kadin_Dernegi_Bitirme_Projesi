import cv2
import numpy as np

# Resmi yükle
image = cv2.imread("uploads/road_narrows_right.jpg")

# Resmin boyutları
height, width = image.shape[:2]

# Resmi HSV renk uzayına dönüştür
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kırmızı renk aralığı
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# Kırmızı pikselleri tespit etmek için maske oluştur
red_mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

# Üst sınırın aralığını genişletmek için bir maske daha oluştur
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
red_mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

# Kırmızı pikselleri kırmızıya dönüştür
red_mask = red_mask1 + red_mask2
image[np.where(red_mask != 0)] = [255, 0, 0]  # Mavi rengi kullan

# Sonucu göster
cv2.imshow('Kırmızı Kısımları Mavi Yapılmış Resim', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


