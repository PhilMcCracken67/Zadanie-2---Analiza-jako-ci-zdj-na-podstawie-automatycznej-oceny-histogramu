import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt


url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

req_obj = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
req = urllib.request.urlopen(req_obj)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

img_bgr = cv2.imdecode(arr, -1)


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Wczytane zdjecie")
plt.axis('off')
fig, axs = plt.subplots(1, 2, figsize=(12, 4))


axs[0].hist(img_gray.ravel(), bins=256, range=[0, 256], color='black')
axs[0].set_title('Histogram - Cale zdjecie (szarosc)')


colors = ('r', 'g', 'b')
for i, col in enumerate(colors):
    axs[1].hist(img_rgb[:, :, i].ravel(), bins=256, range=[0, 256], color=col, alpha=0.6)
axs[1].set_title('Histogram - Kanaly R, G, B')

plt.show()

def assess_quality(image_gray):
    mean_brightness = np.mean(image_gray)
    p5 = np.percentile(image_gray, 5)
    p95 = np.percentile(image_gray, 95)
    contrast = p95 - p5 

    if mean_brightness < 50:
        return "Niedoswietlone (zbyt ciemne)"
    elif mean_brightness > 200:
        return "Przeswietlone (zbyt jasne)"
    elif contrast < 60:
        return "Niski kontrast (mgliste, wyblakle)"
    else:
        return "Dobra jakosc (prawidlowa ekspozycja i kontrast)"

status = assess_quality(img_gray)
print(f"Ocena zdjecia: {status}")
# Poprawa jakoœci 
def improve_image(image_bgr):
   
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    
    limg = cv2.merge((cl, a, b))
    improved_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(improved_bgr, cv2.COLOR_BGR2RGB)

if status != "Dobra jakosc (prawidlowa ekspozycja i kontrast)":
    img_fixed = improve_image(img_bgr)

    plt.figure(figsize=(6,6))
    plt.imshow(img_fixed)
    plt.title("Zdjecie po korekcie histogramu (CLAHE)")
    plt.axis('off')
    plt.show()
