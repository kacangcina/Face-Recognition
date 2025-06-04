import cv2, time
camera = 0
video = cv2.VideoCapture(camera,cv2.CAP_DSHOW)
a = 0
while True:
  a = a + 1
  check, frame = video.read()
  print(check)
  print(frame)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow("tangkap", gray)
  #cv2.waitKey(0)
  key = cv2.waitKey(1)
  if key == ord('q'):
       break
print(a)
video.release()
cv2.destroyAllWindows()