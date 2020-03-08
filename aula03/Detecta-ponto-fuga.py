
import cv2
import numpy as np 

cap = cv2.VideoCapture('teste-1.mp4')

if (cap.isOpened()== False):
    print ("error")

hsv1_M = np.array([1, 0, 240], dtype=np.uint8)
hsv2_M= np.array([150, 20, 255], dtype=np.uint8)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

while (cap.isOpened()):
    ret,frame = cap.read()

    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, hsv1_M, hsv2_M)
        seg = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,np.ones((1, 1)))
        selecao = cv2.bitwise_and(frame, frame, mask=seg)
        blur = cv2.GaussianBlur(selecao,(5,5),0)
        min_contrast = 50
        max_contrast = 250
        linhas = cv2.Canny(blur, min_contrast, max_contrast )
        bordas_color = cv2.cvtColor(linhas, cv2.COLOR_RGB2BGR)

        bc = blur
            

        

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask1,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow ('Frame', bordas_color)


        if cv2.waitKey(25) & 0XFF == ord('q'):
            break

    else:
        break

cap.release()

cv2.destroyAllWindows()
