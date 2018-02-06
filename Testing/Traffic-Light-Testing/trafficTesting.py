from scipy import misc
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imggreen.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

trafficCascade = cv2.CascadeClassifier('traffic.xml')
traffic = trafficCascade.detectMultiScale(grey, 1.1, 2)

for (x,y,w,h) in traffic:
    print("[*] Detected Traffic Light")
    roi_gray = grey[y:y+h, x:x+w]

    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)

    blurred = cv2.GaussianBlur(roi_gray, (41, 41), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)

    print(maxLoc)
    print(maxLoc[0])
    print(maxLoc[1])

    if maxLoc[1] >= 35:
        print("[*] Green Detected")
        lightColor = "Green"
    elif maxLoc[1] <= 30:
        print("[*] Red Detected")
        lightColor = "Red"
    print("[*] Traffic Light Detected")
print("[*] Traffic Light Detection Complete")

plt.imshow(img)
plt.show()
