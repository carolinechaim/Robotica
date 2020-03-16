
import cv2
import numpy as np 

#cap = cv2.VideoCapture('teste-1.mp4')
cap = cv2.VideoCapture('teste-2.mp4')
#cap = cv2.VideoCapture('teste-3.mp4')

if (cap.isOpened()== False):
    print ("error")

hsv1_M = np.array([1, 0, 240], dtype=np.uint8)
hsv2_M= np.array([150, 20, 255], dtype=np.uint8)

# placeholders
aMed_esq = 1
bMed_esq = 1
rhoMed_esq = 1
aMed_dir = 1
bMed_dir = 1
rhoMed_dir = 1

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
    lista_ab = []
    a_esq = []
    b_esq = []
    rho_esq = []
    a_dir = []
    b_dir = []
    rho_dir = []


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
        lines = cv2.HoughLines(linhas, 1, np.pi/180, 150)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                lista_ab.append([a, b, rho])
            for abrho in lista_ab:
                if abrho[0] < 0:
                    a_esq.append(abrho[0])
                    b_esq.append(abrho[1])
                    rho_esq.append(abrho[2])
                elif abrho[0] > 0:
                    a_dir.append(abrho[0])
                    b_dir.append(abrho[1])
                    rho_dir.append(abrho[2])

        if (len(a_esq) &  len(b_esq) & len(rho_esq)) != 0:
            aMed_esq = sum(a_esq) / len(a_esq)
            bMed_esq = sum(b_esq) / len(b_esq)
            rhoMed_esq = sum(rho_esq) / len(rho_esq)

        if (len(a_dir) &  len(b_dir) & len(rho_dir)) != 0:
            aMed_dir = sum(a_dir) / len(a_dir)
            bMed_dir = sum(b_dir) / len(b_dir)
            rhoMed_dir = sum(rho_dir) / len(rho_dir)
        
        x0_esq = aMed_esq*rhoMed_esq
        y0_esq = bMed_esq*rhoMed_esq
        x1_esq = int(x0_esq + 10000*(-bMed_esq))
        y1_esq = int(y0_esq + 10000*(aMed_esq))
        x2_esq = int(x0_esq - 10000*(-bMed_esq))
        y2_esq = int(y0_esq - 10000*(aMed_esq))
        m_e = (y1_esq-y2_esq)/(x1_esq-x2_esq)
        h_esq = int(y1_esq - m_e*x1_esq)
        cv2.line(frame,(x1_esq,y1_esq),(x2_esq,y2_esq),(0,0,255),1)

        x0_dir = aMed_dir*rhoMed_dir
        y0_dir = bMed_dir*rhoMed_dir
        x1_dir = int(x0_dir + 10000*(-bMed_dir))
        y1_dir = int(y0_dir + 10000*(aMed_dir))
        x2_dir = int(x0_dir - 10000*(-bMed_dir))
        y2_dir = int(y0_dir - 10000*(aMed_dir))
        m_d = (y1_dir- y2_dir)/(x1_dir-x2_esq)
        h_dir = int(y1_dir - m_d*x1_dir)
        cv2.line(frame,(x1_dir,y1_dir),(x2_dir,y2_dir),(0,0,255),1)

        
        
        xi = int((h_dir- h_esq)/(m_e-m_d))
        yi = int(m_e*xi + h_esq)

        cv2.circle(frame, (xi,yi), 30, (0,0,225), -1)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask1,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow ('Frame', frame)


        if cv2.waitKey(25) & 0XFF == ord('q'):
            break

    else:
        break

cap.release()

cv2.destroyAllWindows()
