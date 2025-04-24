from ultralytics import YOLO
import cv2
import cvzone
from utils import getPokerHand

cap = cv2.VideoCapture(0)  # Define video to analyse

model = YOLO("Project 4 - Poker Card Identifier/playingCards.pt")

class_names = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S',
               '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C',
               '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD',
               'QH', 'QS']

while True:
    success, img = cap.read()
    results = model(img)

    hand = []
    for r in results:
        for box in r.boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            conf = round(box.conf[0].item(), 2)
            card_label = class_names[int(box.cls[0])]
            if conf > 0.5:
                hand.append(card_label)

            cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0), thickness=3)
            cv2.putText(img, text=f"{card_label}, {conf}", org=(max(0,x1), max(35,y1-15)), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=(0,0,0), thickness=3)
    
    hand = list(set(hand))
    if len(hand) == 5:
        hand_name = getPokerHand(hand)
        cvzone.putTextRect(img, text=f"Your Hand : {hand_name}", pos=(25, 50), scale=3, thickness=3, colorT=(0,0,0), colorR=(38,133,0))
    
    cv2.imshow("Hand Identifier", img)
    cv2.waitKey(1)
            
    
