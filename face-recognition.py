import cv2


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("test")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors =4,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", img)
cv2.waitKey(0)
