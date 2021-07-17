import cv2

// auther : ohood 
cascade_classifier = cv2.CascadeClassifier('Face Detect/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

   

    # To Display the resulting Frame
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# To release the capture
cap.release()
cv2.destroyAllWindows()