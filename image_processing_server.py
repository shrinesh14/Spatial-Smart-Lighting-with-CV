import torch
import cv2
import numpy as np
import time
import warnings
import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import requests
import time
warnings.filterwarnings("ignore", category=FutureWarning)



yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform

midas = midas.cpu()
yolo_model = yolo_model.cpu()
midas.eval()

cap = cv2.VideoCapture(1)

url = 'http://localhost:3000/sensors'
data = {}
classroom = "AB1-202"
people_count = 0
near_depth = None
far_depth = None
ceiling_depth = None
ceiling_start_depth = None
new_point_depth = None
click_position = None
ceiling_point = None
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global click_position, mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        mouse_clicked = True

# Set up mouse callback for marking ceiling points
cv2.namedWindow('YOLOv5 + MiDaS')
cv2.setMouseCallback('YOLOv5 + MiDaS', mouse_callback)

def get_depth_at_point(depth_map, x, y):
    """ Get depth value at the specified point in the depth map. """
    if x < depth_map.shape[1] and y < depth_map.shape[0]:
        return depth_map[y, x]
    return None

def find_relative_depth(value,near_depth,far_depth,celing_start_depth,celing_depth):
    val = ((value-far_depth)/(near_depth-far_depth))*(celing_start_depth-celing_depth) + celing_depth
    return(val)

while not mouse_clicked:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 detection
    results = yolo_model(frame)
    detections = results.pred[0].cpu().numpy()  # Move to CPU before converting to NumPy

    # Prepare the image for MiDaS depth estimation
    img_midas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_midas = midas_transforms(img_midas).to('cpu')

    # Estimate the depth map
    with torch.no_grad():
        depth_map = midas(img_midas)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert depth map to NumPy
    depth_map = depth_map.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    for *box, conf, class_id in detections:
        if int(class_id) == 0:  # Person class
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            near_depth = get_depth_at_point(depth_map, center_x, center_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Near Depth: {near_depth:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('YOLOv5 + MiDaS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if click_position:
    ceiling_x, ceiling_y = click_position
    far_depth = get_depth_at_point(depth_map, ceiling_x, ceiling_y)
mouse_clicked = False
time.sleep(1)

while not mouse_clicked:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = results.pred[0].cpu().numpy()

    img_midas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_midas = midas_transforms(img_midas).to('cpu')

    with torch.no_grad():
        depth_map = midas(img_midas)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = depth_map.cpu().numpy()

    for *box, conf, class_id in detections:
        if int(class_id) == 0:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            far_depth = get_depth_at_point(depth_map, center_x, center_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Far Depth: {far_depth:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('YOLOv5 + MiDaS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if click_position:
    ceiling_x, ceiling_y = click_position
    far_depth = get_depth_at_point(depth_map, ceiling_x, ceiling_y)
mouse_clicked = False
time.sleep(1)
print("Click celing end")
while not mouse_clicked:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, 'Click celing end', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv5 + MiDaS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if click_position:
    ceiling_x, ceiling_y = click_position
    ceiling_depth = get_depth_at_point(depth_map, ceiling_x, ceiling_y)

mouse_clicked = False
print("Click celing start")
while not mouse_clicked:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, 'Click celing start', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv5 + MiDaS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if click_position:
    ceiling_x, ceiling_y = click_position
    ceiling_start_depth = get_depth_at_point(depth_map, ceiling_x, ceiling_y)
mouse_clicked = False
new_point_coor = []
new_point_depth = []
run = True
person_depth = None
while cap.isOpened() and run:
    data = {}
    data["LABEL"] = classroom
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo_model(frame)
    detections = results.pred[0].cpu().numpy()

    img_midas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_midas = midas_transforms(img_midas).to('cpu')
    image_mask = np.zeros((height, width, 3), np.uint8)
    # Estimate the depth map
    with torch.no_grad():
        depth_map = midas(img_midas)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = depth_map.cpu().numpy()
    people_count = 0
    if mouse_clicked:
        new_point_depth.append(get_depth_at_point(depth_map, click_position[0], click_position[1]))
        new_point_coor.append([(click_position[0],click_position[1])])
        
        mouse_clicked = False 

    for *box, conf, class_id in detections:
        if int(class_id) == 0:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            person_depth = get_depth_at_point(depth_map, center_x, center_y)
            person_depth = find_relative_depth(person_depth,near_depth,far_depth,ceiling_start_depth,ceiling_depth)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mid = (x1+x2)/2
            people_count = people_count +1 
            x_vals = []
            dist = []
            
            if(person_depth):
                for i in range(len(new_point_depth)):
                    if new_point_depth[i] is not None and ceiling_depth is not None:
                        distance_from_ceiling_to_human = abs(new_point_depth[i] - person_depth)
                        dist.append(distance_from_ceiling_to_human)
                        x_value = abs(mid-new_point_coor[i][0][0])
                        x_vals.append(x_value)
                        # cv2.circle(frame, new_point_coor[i][0], 5, (0,255,0), thickness=-1)
                        # cv2.circle(image_mask, new_point_coor[i][0], 5, (255,0,0), thickness=-1)
                    x_max = 100
                    z_max = 140
                    if(len(new_point_coor) > 0):
                        for i in range(0,len(new_point_coor)):
                            cv2.circle(frame, new_point_coor[i][0], 5, (0,255,0), thickness=-1)
                            cv2.circle(image_mask, new_point_coor[i][0], 5, (255,0,0), thickness=-1)
                    for i in range(len(x_vals)):
                        data[f"L{i+1}"]  = False
                    for i in range(len(x_vals)):
                        if(abs(dist[i]) <= abs(ceiling_depth*(z_max/100))):
                            if(abs(x_vals[i]) <= (width*(x_max/100))):
                                cv2.circle(image_mask, new_point_coor[i][0], 5, (0,255,0), thickness=-1)
                                data[f"L{i+1}"]  = True
                            elif(data[f"L{i+1}"]  != True):
                                data[f"L{i+1}"]  = False
                        elif(data[f"L{i+1}"]  != True):
                            data[f"L{i+1}"]  = False
        data["peopleCount"] = people_count
        # indexed_list = list(enumerate(x_vals))
        # sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
        # n = int(len(x_vals)/2)
        # top_least_indices = [index for index, value in sorted_indexed_list[:n]]
        # least_val_ind = 0
        # for i in top_least_indices:
        #     if(dist[i] < dist[least_val_ind]):
        #         least_val_ind = i
        # cv2.circle(image_mask, new_point_coor[least_val_ind][0], 5, (0,255,0), thickness=-1)
    cv2.imshow('YOLOv5 + MiDaS', frame)
    cv2.imshow('Light Mask', image_mask)
    try:
        response = requests.post(url, json=data)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()