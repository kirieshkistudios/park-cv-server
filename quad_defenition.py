# Part 1: Save Parking Space Coordinates (quad_definition.py)
import cv2
import json
import numpy as np

def mouse_handler(event, x, y, flags, params):
    img, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0,0,255), -1)
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (0,255,0), 2)
            if len(points) == 4:
                cv2.polylines(img, [np.array(points)], True, (0,255,0), 2)
            cv2.imshow('Parking Space Definition', img)

def save_parking_space(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    points = []
    cv2.namedWindow('Parking Space Definition')
    cv2.setMouseCallback('Parking Space Definition', mouse_handler, (img, points))
    
    while True:
        cv2.imshow('Parking Space Definition', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) == 4:
            break
    
    if len(points) == 4:
        with open('parking_space.json', 'w') as f:
            json.dump(points, f)
        print("Parking space coordinates saved")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_parking_space("parking_ref.jpg")