# Import library yang diperlukan
from scipy.spatial import distance as dist
from collections import deque
import numpy as np
import argparse
from imutils import perspective
import imutils
import cv2
import math
import urllib #Untuk membaca Gambar dari URL
import serial

# Insisalisasi koneksi serial untuk komunikasi dengan perangkat eksternal
ser1 = serial.Serial('/dev/cu.usbmodem1D11401', 9600)

# Fungsi untuk menghitung titik tengah antara dua titik
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Parsing argumen dari basis perintah
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# Inisialisasi lebar objek
width = 7.87 # (inch)

# Definsi rentang warna untuk deteksi objek
lower = {
    'red': (0, 127, 90),
    'green': (62, 82, 0),
    'blue': (90, 150, 140),
    'yellow': (23, 59, 119),
    'orange': (0, 130, 190),
}
lower['blue'] = (93, 10, 0)

upper = {
    'red': (5, 255, 255),
    'green': (99, 255, 245),
    'blue': (102, 255, 255),
    'yellow': (54, 255, 255),
    'orange': (5, 255, 255)
}

# Definisi warna standar untuk lingkaran di sekitar objek
colors = {
'red':(179,255,255), 
'green':(109,255,255), 
'blue':(255,0,0), 
'yellow':(0, 255, 217), 
'orange':(0,140,255)
}
 
# Jika jalur video tidak disediakan, ambil referensi ke webcam
if not args.get("video", False):
    camera = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 ", cv2.CAP_GSTREAMER)
# Jika tidak, ambil referensi ke file video
else :
    camera = cv2.VideoCapture(args["video"])

nilaitengah = str(320)
tolerance_value = 30
pointsList = []

#Loop utama
while True:

    # Pengaturan parameter kamera
    camera.set(28, 255)
    (grabbed, frame) = camera.read()
 
    # Preprocessing citra
    frame = imutils.resize(frame, width=640, height=960)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Merah
    kernel1 = np.ones((9,9), np.uint8)
    mask_red = cv2.inRange(hsv, lower['red'], upper['red']) 
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel1)
    cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # Hijau
    kernel2 = np.ones((9,9), np.uint8)
    mask_green = cv2.inRange(hsv, lower['green'], upper['green'])
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel2)
    cnts_green = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Penggambaran warna dan objek pada frame
    col = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255), (255,255,255))
    refObj_red = None
    refObj_green = None
    box = None
    box_green = None
    box_red = None
    
    testt = frame.copy()

    # Jika setidaknya satu kontur ditemukan
    if len(cnts_green) > 0:
        c = max(cnts_green, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
        # Jika radiusnya memenuhi ukuran minimum.
        if radius > 0.5:
            cv2.circle(testt, (int(x), int(y)), int(radius), colors["green"], 2)
            cv2.putText(testt,"green ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors["green"],2)

    for c in cnts_green:

        # Jika konturnya kurang besar, abaikan saja
        if cv2.contourArea(c) < 100:
            continue
        # Menghitung kotak pembatas yang diputar dari kontur
        box_green = cv2.minAreaRect(c)
        box_green = cv2.cv.BoxPoints(box_green) if imutils.is_cv2() else cv2.boxPoints(box_green)
        box_green = np.array(box_green, dtype="int")

        # Inisialisasi objek referensi jika belum ada
        if refObj_green is None :
            box_green = perspective.order_points(box_green)

            # Menghitung titik tengah kotak pembatas
            cX_green = np.average(box_green[:, 0])
            cY_green = np.average(box_green[:, 1])

            # referensi objek
            (tl, tr, br, bl) = box_green
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Hitung jarak Euclidean antara titik tengah, lalu buat objek referensi
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj_green = (box_green, (cX_green, cY_green), D / width)
            continue

        # Gambar kontur dalam sebuah gambar(image)
        cv2.drawContours(testt, [box_green.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(testt, [refObj_green[0].astype("int")], -1, (0, 255, 0), 2)
        

    if len(cnts_red) > 0:

        # Mencari kontur terbesar pada masker, lalu gunakan untuk menghitung lingkaran penutup minimum dan pusatroid
        c = max(cnts_red, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
        # Lanjutkan jika radiusnya memenuhi ukuran minimum.
        if radius > 0.5:

            # Gambar lingkaran dan pusat massa pada bingkai, lalu perbarui daftar titik yang dilacak
            cv2.circle(testt, (int(x), int(y)), int(radius), colors["red"], 2)
            cv2.putText(testt,"Red ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors["red"],2)
    

    for c in cnts_red:

        # Jika konturnya kurang besar, abaikan saja
        if cv2.contourArea(c) < 100:
            continue

        # Menghitung kotak pembatas yang diputar dari kontur
        box_red = cv2.minAreaRect(c)
        box_red = cv2.cv.BoxPoints(box_red) if imutils.is_cv2() else cv2.boxPoints(box_red)
        box_red = np.array(box_red, dtype="int")
        box_red = perspective.order_points(box_red)
        cX_red = np.average(box_red[:, 0])
        cY_red = np.average(box_red[:, 1])

        if refObj_red is None:

            # jika ini adalah kontur pertama yang kita periksa (yaitu, kontur paling kiri), anggap ini adalah objek referensi
            (tlr, trr, brr, blr) = box_red
            (tlrblrX, tlrblrY) = midpoint(tlr, blr)
            (trrbrrX, trrbrrY) = midpoint(trr, brr)

            # Hitung jarak Euclidean antara titik tengah, lalu buatlah objek referensi
            D = dist.euclidean((tlrblrX, tlrblrY), (trrbrrX, trrbrrY))
            refObj_red = (box_red, (cX_red, cY_red), D / width)
            continue
        
        # Gambar kontur pada gambar (image)
        cv2.drawContours(testt, [box_red.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(testt, [refObj_red[0].astype("int")], -1, (0, 255, 0), 2)
    
    if refObj_green != None and refObj_red != None and box_green.any() and box_red.any():

        # Menumpuk koordinat referensi dan koordinat objek untuk memasukkan pusat objek
        refCoords_red = np.vstack([refObj_red[0], refObj_red[1]])
        refCoords_green = np.vstack([refObj_green[0], refObj_green[1]])
        objCoords_red = np.vstack([box_red, (cX_red, cY_red)])
        objCoords_green = np.vstack([box_green, (cX_green, cY_green)])
 
        # Koordinat titik pusat kotak pembatas
        xA = refCoords_red[4][0]
        yA = refCoords_red[4][1]
        xB = objCoords_green[4][0]
        yB = objCoords_green[4][1]
        color = col[5]

        # Menggambar lingkaran yang sesuai dengan titik-titik saat ini dan menghubungkannya dengan sebuah garis
        cv2.line(testt, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)

        # Menghitung jarak Euclidean antara koordinat, lalu ubah jarak dalam piksel menjadi jarak dalam satuan
        D = dist.euclidean((xA, yA), (xB, yB)) / refObj_green[2]
        (mX, mY) = midpoint((xA, yA), (xB, yB))
        nilaitengah = str(mX)

        cv2.putText(testt, "{:.1f} cm".format(D * 2.54), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        iA = xA
        jA = yA
        iB = xB
        jB = yB

        # Menggambar garis pov_camera
        cv2.line(testt, (int(camera.get(3)/2), int(camera.get(4))), (int(iA/2) + int(iB/2),int(jA/2) + int(jB/2)), color, 2)

        # Menggambar garis horizontal_line
        horizontal_line = cv2.line(testt, (int(camera.get(3)/2), int(0)), (int(camera.get(3)/2), int(camera.get(4))), col[1], 2)

        
        dot_pov_camera = cv2.circle(testt, (int(iA/2) + int(iB/2), int(jA/2) + int(jB/2)), 5, color, -1)
        pointsList.append([int(iA/2) + int(iB/2), int(jA/2) + int(jB/2)])
        
        top_dot_horizontal_line = cv2.circle(testt, (int(camera.get(3)/2), int(0)), 5, col[1], -1)
        pointsList.append([int(camera.get(3)/2), int(0)])

        bottom_dot_horizontal_line = cv2.circle(testt, (int(camera.get(3)/2), int(camera.get(4))), 5, col[1], -1)
        pointsList.append([int(camera.get(3)/2), int(camera.get(4))])

        #hampir bener tapi mentok di 90 trs sudutnya cuma setengah dari sudut asli di dunia nyata
        def getAngle():
            dot_pov_camera, bottom_dot_horizontal_line, top_dot_horizontal_line = pointsList[-3:] 
            m1=(((int(iA/2) + int(iB/2))-int(camera.get(3)/2))/((int(jA/2) + int(jB/2))-int(camera.get(4))))
            srad=math.atan(m1)
            sdeg=math.degrees(srad)

            if sdeg < 0:
                ser1.write(sdeg)

            elif sdeg > 0:
                ser1.write(sdeg)

            else:
                print("No need to move horizontally")          
        
        if len(pointsList) % 3 == 0 : 
            getAngle()
        
    cv2.imshow("Frame", testt)
    
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
