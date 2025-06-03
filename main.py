import cv2
from ultralytics import YOLO
import serial
import time
import numpy as np
model = YOLO("Weights\\best.pt")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cap = cv2.VideoCapture(0)
#ser=serial.Serial('COM3',115200,timeout=1)
time.sleep(2)

msr_sigma=[15,80,250]
msr_weight=[1/3,1/3,1/3]

def multiscaleRetinex(img,sigmas,weights=None):
    if weights is None:
        weights=[1.0/len(sigmas)]*len(sigmas)
    img= img+1.0
    log_img=np.log10(img)
    retinex=np.zeros_like(log_img)
    for sigma, w in zip(sigmas,weights):
        blurred=cv2.GaussianBlur(img(0,0),sigmaX=sigma,sigmaY=sigma)
        log_blurred=np.log10(blurred+1.0)
        retinex += w*(log_img-log_blurred)
    return retinex
def normalize_retinex(retinex_image):
    min_val=np.min(retinex_image)
    max_val=np.max(retinex_image)
    if max_val-min_val<1e-6:
        scaled=np.zeros_like(retinex_image)
    else:
        scaled=(retinex_image-min_val)/(max_val-min_val)*255.0

    return np.uint8(np.clip(scaled,0,255))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    denoised=cv2.fastNlMeansDenoisingColored(frame,None,h=10,hColor=10,templateWindowSize=7,searchWindowSize=21)
    img_float=denoised.astype(np.float32)
    retinex_result=multiscaleRetinex(img_float,msr_sigma,msr_weight)
    msr_normalized=normalize_retinex(retinex_result)

    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl    = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    input_size=(640,640)
    resized=cv2.resize(enhanced_frame,input_size,interpolation=cv2.INTER_LINEAR)

    results = model(enhanced_frame,conf=0.5)[0]  

    num_objects = len(results.boxes)

    #if num_objects >= 3:
    #    ser.write(b"Collect")

   
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

   
    cv2.putText(enhanced_frame, f"Count: {num_objects}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  
    cv2.imshow("YOLOv8n + CLAHE", enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
