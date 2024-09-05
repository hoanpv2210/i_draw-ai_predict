import cv2
import mediapipe as mp
import numpy as np
#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
import joblib
import os
import torch
from model_quick_draw import quick_draw_CNN
#cai dat
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity =0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
cap = cv2.VideoCapture(0) # khoi tao camera
ten = ["apple","book","bowtie","candle","cup","door","envelope","eyeglass","guitar","hammer","hat","ice cream","leaf","pants","scissors","star","t-shirt"]
class Main():
    def __init__(self):
        self.cx_min = 640
        self.cy_min = 480
        self.cx_max = 0
        self.cy_max = 0
        self.points = []
        self.xu_ly  = 0
    def max_min_x_y(self,cx,cy):
        if self.cx_min > cx: self.cx_min = cx
        if self.cy_min > cy: self.cy_min = cy
        if self.cx_max < cx: self.cx_max = cx
        if self.cy_max < cy: self.cy_max = cy
    def print_line(self,img,point):
        if len(point) > 1:
                    for i in range(len(point) - 1):
                        cv2.line(img, point[i], point[i + 1], (0, 0, 0), 2)
    def detect_hold_hand(self,hand_landmark):
        MIDDLE_TIP = hand_landmark.landmark[mp_hand.HandLandmark.MIDDLE_FINGER_TIP]
        MIDDLE_MCP = hand_landmark.landmark[mp_hand.HandLandmark.MIDDLE_FINGER_MCP]
        distance = ((MIDDLE_TIP.x-MIDDLE_MCP.x)**2+(MIDDLE_TIP.y-MIDDLE_MCP.y)**2)**0.5
        if distance >0.1:
            return True
        else:    
            return False
    def creat_image_print(self):
        #cv2.rectangle(self.img, (self.cx_min,self.cy_min),(self.cx_max,self.cy_max),(0,255,0),2)
        if (self.cx_max - self.cx_min) >= (self.cy_max - self.cy_min):
            #kc = int((self.cx_max - self.cx_min - (self.cy_max - self.cy_min))/2)
            #roi_gray = self.img[(self.cy_min-kc):(self.cy_max+kc), self.cx_min:self.cx_max]
            img_creat = np.ones(((self.cx_max - self.cx_min), (self.cx_max - self.cx_min)), dtype=np.uint8) * 255
        else:
            img_creat = np.ones(((self.cy_max - self.cy_min), (self.cy_max - self.cy_min)), dtype=np.uint8) * 255
            #kc = int((self.cy_max - self.cy_min - (self.cx_max - self.cx_min))/2)
            #roi_gray = self.img[self.cy_min:self.cy_max, (self.cx_min-kc):(self.cx_max+kc)]
        return img_creat
    def detect_hand(self):
        if self.result.multi_hand_landmarks:
            for idx, hand in enumerate(self.result.multi_hand_landmarks):
                lbl = self.result.multi_handedness[idx].classification[0].label
                if lbl == "Right":
                    for id, lm in enumerate(hand.landmark):
                        h,w, _ = self.img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        if id ==8:
                            cv2.circle(self.img, (cx,cy),10,(0, 255, 0),cv2.FILLED)
                            self.points.append((cx,cy))
                            self.max_min_x_y(cx,cy)
                        if not self.detect_hold_hand(hand) and self.xu_ly ==0:
                            img_creat = self.creat_image_print()
                            if (self.cx_max - self.cx_min) >= (self.cy_max - self.cy_min):
                                kk = int((self.cx_max - self.cx_min -(self.cy_max - self.cy_min))/2)
                                point_creat = [(x - self.cx_min, (y - self.cy_min + kk)) for (x, y) in self.points]
                            else:
                                kk = int((self.cy_max - self.cy_min -(self.cx_max - self.cx_min))/2)
                                point_creat = [((x - self.cx_min + kk), y - self.cy_min) for (x, y) in self.points]
                            self.print_line(img_creat,point_creat)
                            cv2.imwrite('output_with_lines.jpg', img_creat)
                            img_creat = cv2.resize(img_creat, (28,28))
                            img_creat = img_creat.astype(np.float32)
                            img_creat = (255 - img_creat)/255
                            img_creat = np.array(img_creat, dtype=np.float32)[None, None, :, :]
                            img_creat =  torch.from_numpy(img_creat)
                            with torch.no_grad():
                                y_pred = model(img_creat)
                                y_classes = [np.argmax(element) for element in y_pred]
                                self.a = y_classes[0]
                            #cv2.imwrite('output_with_lines.jpg', img_creat)
                            self.xu_ly = 1
                        cv2.putText(self.img, ten[self.a], (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        if self.xu_ly == 1 and  self.detect_hold_hand(hand):
                            y_classes = []
                            self.xu_ly = 0
                            self.points = []
                            self.cx_min = 640
                            self.cy_min = 480
                            self.cx_max = 0
                            self.cy_max = 0
        self.print_line(self.img,self.points)
    def read_write_data(self):
        self.a = 0
        while cap.isOpened():
            cam_ok, self.img = cap.read()
            self.result = hands.process(self.img)
            if not cam_ok:
                break
            self.detect_hand()
            cv2.imshow("Nhan dang ban tay", self.img)
            key = cv2.waitKey(1)
            if key == 27:
                break
# def process_data():
#     (X_train, y_train), (X_test,y_test) = datasets.mnist.load_data()
#     X_train = X_train / 255.0
#     X_test = X_test / 255.0
#     cnn = models.Sequential([
#     layers.Conv2D(filters=28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28,1)),
#     layers.MaxPooling2D((2, 2)),
    
#     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
#     ])
#     cnn.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#     cnn.fit(X_train, y_train, epochs=10)
#     return cnn
    
if __name__ == "__main__":
    #cnn = process_data()
    #clf = joblib.load('saved_model_1.pkl')
    checkpoint = torch.load(os.path.join("trained_models_quick_draw","last.pt"))
    model = quick_draw_CNN()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    main_instance = Main()
    main_instance.read_write_data()
    cap.release()
    cv2.destroyAllWindows()