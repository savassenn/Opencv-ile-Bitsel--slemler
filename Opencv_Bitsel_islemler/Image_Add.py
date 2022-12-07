import cv2
import numpy as np

# print(f16.shape) #  403,720
# print(logo.shape) # 136,205

f16 = cv2.imread("f16.jpg")
logo = cv2.imread("logo.jpg")

y,x,z = logo.shape
roi = f16[0:y,260:465]

logo_gri = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gri,150,255,cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

f16_arka = cv2.bitwise_and(roi,roi,mask = mask)
logo_arka = cv2.bitwise_and(logo,logo, mask = mask_inv)
toplam = cv2.add(f16_arka,logo_arka)

f16[0:y,260:465] = toplam

cv2.imshow("F16",f16)
# cv2.imshow("LOGO",logo)
# cv2.imshow("LOGO_GRI",logo_gri)
# cv2.imshow("MASK",mask)
# cv2.imshow("MASK_INV",mask_inv)
# cv2.imshow("ROI",roi)
# cv2.imshow("F16_ARKA",f16_arka)
# cv2.imshow("LOGO_ARKA",logo_arka)
#cv2.imshow("TOPLAM",toplam)



cv2.waitKey(0)
cv2.destroyAllWindows()