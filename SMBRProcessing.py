import cv2
import numpy as np
import csv
##print(cv2.__version__)
print("numpy and opencv successfuly imported")

# Read in AVI file:
# Create a VideoCapture object via passing in filename. 
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture
filename = "robot_moving_5_crop.avi"
cap = cv2.VideoCapture("/Users/aliristang/Desktop/robot_moving_5_crop.avi") #try home if Users doesn't work

#init variables 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Window setup
font = cv2.FONT_HERSHEY_SIMPLEX
frame_rate = round(cap.get(cv2.CAP_PROP_FPS), 2) # get the frame rate of the video, should be 10fps
xdim = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # get the frame width 
ydim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # get the frame height
print("Image Width: %d pixels" % xdim)
print ("Image Height: %d pixels" % ydim)
print ("Frame Rate: %.2f fps" % frame_rate)

# Grab the first frame of the video 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame0 = cap.read() # Read the frame
frame0Neg = cv2.bitwise_not(frame0)
clone_frame0 = frame0.copy() # Copy the frame
frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
#cv2.putText(frame0,'Frame: %d' % frame_no,(10,50), font,1,(255,255,255),2,cv2.LINE_AA)
#cv2.imshow('frame', frame0) no need to show frame
cv2.waitKey(2000) #in milliseconds

##### SIMPLE BLOB DETECTOR CODE
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
# # Change thresholds
# params.minThreshold = 0
# params.maxThreshold = 50

# # Set up the detector with default parameters.
# print("init blobDetector")
# detector = cv2.SimpleBlobDetector_create(params)

# # Detect blobs, .detect() returns list of Blob objects (should be 1 in this case)
# print("detecting blobs")
# blob = detector.detect(frame0)

# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# print("supposedly drawing blobs")
# im_with_keypoints = cv2.drawKeypoints(frame0, blob, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Show keypoints
# # cv2.imshow("KeypointsNeg", im_with_keypointsNeg)
# cv2.imshow("First Frame Check", im_with_keypoints)

##### CONTOUR DETECTION OF FIRST FRAME
lower = (0,0,0) 
upper = (10,10,10) 
mask = cv2.inRange(frame0, lower, upper)
try:
    # NB: using _ as the variable name for two of the outputs, as they're not used
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blob = max(contours, key=lambda el: cv2.contourArea(el))
    M = cv2.moments(blob)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    x = str(int(M["m10"] / M["m00"]))
    y = str(int(M["m01"] / M["m00"]))
    print("x: " + x + ", y: " + y)

    cv2.circle(frame0, center, 2, (0,0,255), -1)

except (ValueError, ZeroDivisionError):
    pass

cv2.imshow('frame',frame0)
cv2.imshow('canvas',frame0)
cv2.imshow('mask',mask)

cv2.waitKey(0)



####################### DETECTING FIRST FRAME DONE, PROCESS WHOLE VIDEO BELOW

# init final output arrays r
xpos = []
ypos = []
num_frames = []
time_stamp = []
cap.set(cv2.CAP_PROP_POS_FRAMES,0);

processAll = input("Do you want to process all video frames? [y/n]: ")

if processAll.startswith('y'):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    frame_count = int(input("Enter the index of the last frame you want to process: "))

save_vid = input("Do you to save the processed video? [y/n]")
if save_vid.startswith('y'):
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out = cv2.VideoWriter('%s_processed.mp4' % filename,fourcc, frame_rate, (xdim,ydim), 1)


i = 0 # set up frame counter
print ("Processing Full Video...")
while(cap.isOpened()):
    time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 # current position of the video file in seconds
    k = cap.get(cv2.CAP_PROP_POS_FRAMES)
    num_frames.append(k)
    time_stamp.append(round(time,2))

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_clone = frame.copy() # Copy the frame

    ##### detect blob, find center position, store info in arr, display frame
    lower = (0,0,0) 
    upper = (10,10,10) 
    mask = cv2.inRange(frame, lower, upper)
    try:
        # NB: using _ as the variable name for two of the outputs, as they're not used
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blob = max(contours, key=lambda el: cv2.contourArea(el))
         
        M = cv2.moments(blob)
        center_x = (int(M["m10"] / M["m00"]))
        center_y = (int(M["m01"] / M["m00"]))

        x = str(center_x)
        y = str(center_y)
        print("x: " + x + ", y: " + y)

        xpos.append(center_x)
        ypos.append(center_y)

        cv2.putText(frame_clone,'Frame: %d' % k,(10,50), font,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame_clone,'Time: %.2f' % time,(10,100), font,1,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(frame_clone, center, 2, (0,0,255), -1)

    except (ValueError, ZeroDivisionError):
        pass

    # Display the resulting frame
    cv2.imshow('frame', frame_clone)
    cv2.imshow('mask', mask)
    if save_vid.startswith('y'):
        out.write(frame_clone)
    #cv2.imshow('frame',frame0)
    #cv2.imshow('canvas',frame0)

    cv2.waitKey(0)

    i = i+1 #increase frame count
    if i == frame_count:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Done Video Processing.")

filename = 'smbr_opencv_result.csv'
print("x0): " + xpos[0])
print("y0): " + ypos[0])

with open(filename, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(test_arr)
    writer.writerows([["Date",now.strftime("%Y/%m/%d")]])
    writer.writerows([["Sample",filename]])
    writer.writerows([["Frame Rate",frame_rate]])
    writer.writerows([["Average Velocity", avg_vel_pix, "pixels/s"]])
    #writer.writerows([["Average Velocity", avg_vel_mm, "mm/s"]])
    writer.writerows([["Camera Matrix",K]])
    writer.writerows([["Frame_No", "Time", "px","py", "mx", "my", "mz"]])
    print("Done Writing to CSV.")

        




    
