# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
#from google.colab.patches import cv2_imshow

# part 1:

def load_obj_each_frame(data_file):
  with open(data_file, 'r') as file:
    frame_dict = json.load(file)
  return frame_dict

def draw_target_object_center(video_file,obj_centers):
  count = 0
  cap = cv.VideoCapture(video_file)
  frames = []
  ok, image = cap.read()
  vidwrite = cv.VideoWriter("part_1_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
  while ok:
    pos_x,pos_y = obj_centers[count]
    count+=1
    ######!!!!#######
    image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
    ######!!!!#######
    image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)
    vidwrite.write(image)
    ok, image = cap.read()
  vidwrite.release()

def compute_fps(video):   #get the frames per second
  cap = cv.VideoCapture(video)
  fps = cap.get(cv.CAP_PROP_FPS)
  return fps

def kalman_filter(obj_centers, video):
  cap = cv.VideoCapture(video)
  fps = compute_fps(video)
  ok, image = cap.read()
  dt = 1/fps    #time steps
  cov_x = 1.0001
  cov_y = 1.1

  #state transition matrix
  F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

  #initial state covariance matrix 
  P = np.eye(4)

  #process noise covariance matrix
  # Q = np.array([[dt**4/4, 0, dt**3/3, 0],
  #               [0, dt**4/4, 0, dt**3/3],
  #               [dt**3/3, 0, dt**2/2, 0],
  #               [0, dt**3/3, 0 , dt**2/2]])
  Q = np.eye(4) * 0.1

  #measurement covariance matrix
  R = np.array([[cov_x**2, 0],
              [0, cov_y**2]])

  #measurement matrix 
  H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
  
  #initial states
  x_k = np.array([[obj_centers[1][0]],
                [obj_centers[1][1]],
                [0],
                [0]])

  #control matrix
  # B = np.array([[0], [0], [0], [0]])
  # u_k = acceleration

  i = 0
  total_count = 0
  estimated_centers = []
  while ok:
      pos_x, pos_y = obj_centers[i]
      image = cv.resize(image, (700, 500))

      #prediction
      x_k_hat = F @ x_k 
      P_hat = F @ P @ F.T + Q
      print(x_k_hat)

      if (pos_x !=-1 and pos_y != -1):
          K_prime = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + R)
          z_k = np.array([[pos_x], 
                        [pos_y]])
          x_k = x_k_hat + K_prime @ (z_k - H @ x_k_hat)
          P = (np.eye(4) - K_prime @ H) @ P_hat
          total_count += 1
      else:
          x_k = x_k_hat
          P = P_hat

      pred_x = int(x_k_hat[0])
      pred_y = int(x_k_hat[1])

      estimated_centers.append([pred_x, pred_y])
      i += 1
      ok, image = cap.read()
  return estimated_centers


def draw_kf_estimated_centers(video_file, estimated_centers):
  cap = cv.VideoCapture(video_file)
  ok, image = cap.read()
  count = 0
  vidwrite = cv.VideoWriter("car_tracking_kf.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
  all_positions = []
  while ok:
      pos_x, pos_y = estimated_centers[count]

      count += 1
      image = cv.resize(image, (700, 500))
      image = cv.circle(image, (int(pos_x), int(pos_y)), 1, (0, 0, 255), 2)
      all_positions.append((int(pos_x), int(pos_y)))
      if len(all_positions) > 0:
          for i in range(1, len(all_positions)):
              cv.line(image, all_positions[i-1], all_positions[i], (255, 0, 0), 2)   #connecting prev point with current point

      vidwrite.write(image)
      ok, image = cap.read()
  vidwrite.release()

frame_dict = load_obj_each_frame("part_1_object_tracking.json")
video_file = "commonwealth.mp4"
#draw_target_object_center(video_file,frame_dict['obj'])
estimated_centers = kalman_filter(frame_dict['obj'], video_file)
draw_kf_estimated_centers(video_file, estimated_centers)

# part 2:

def draw_object(object_dict,image,color = (0, 255, 0), thickness = 2,c_color= \
                (255, 0, 0)):
  # draw box
  x = object_dict['x_min']
  y = object_dict['y_min']
  width = object_dict['width']
  height = object_dict['height']
  image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)
  return image

def draw_objects_in_video(video_file,frame_dict):
  count = 0
  cap = cv.VideoCapture(video_file)
  frames = []
  ok, image = cap.read()
  vidwrite = cv.VideoWriter("part_2_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
  while ok:
    ######!!!!#######
    image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
    ######!!!!#######
    obj_list = frame_dict[str(count)]
    for obj in obj_list:
      image = draw_object(obj,image)
    vidwrite.write(image)
    count+=1
    ok, image = cap.read()
  vidwrite.release()

frame_dict = load_obj_each_frame("part_2_frame_dict.json")
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file,frame_dict)

