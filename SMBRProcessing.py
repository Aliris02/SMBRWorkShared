import cv2
import numpy as np
##print(cv2.__version__)
print("numpy and opencv successfuly imported")

# Read in AVI file:
# Create a VideoCapture object via passing in filename. 
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture
filename = "robot_moving_5_crop.avi"
cap = cv2.VideoCapture("/Users/aliristang/Desktop/robot_moving_5_crop.avi") #try home if Users doesn't work


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
cv2.putText(frame0,'Frame: %d' % frame_no,(10,50), font,1,(255,255,255),2,cv2.LINE_AA)
#cv2.imshow('frame', frame0) no need to show frame
cv2.waitKey(3000) #wait is in milliseconds

print("made it past waitKey")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 50

# Set up the detector with default parameters.
print("init blobDetector")
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
print("detecting blobs")
keypoints = detector.detect(frame0)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
print("supposedly drawing blobs")
im_with_keypoints = cv2.drawKeypoints(frame0, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
# cv2.imshow("KeypointsNeg", im_with_keypointsNeg)
cv2.imshow("First Frame Check", im_with_keypoints)

cv2.waitKey(0)

####################### DETECTING FIRST FRAME DONE, PROCESS WHOLE VIDEO BELOW

i = 0 # set up frame counter
print ("Processing Full Video...")
while(cap.isOpened()):
    time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 # current position of the video file in seconds
    k = cap.get(cv2.CAP_PROP_POS_FRAMES)
    num_frames.append(k)
    time_stamp.append(round(time,2))

    # Capture frame-by-frame
    ret, frame = cap.read()

    clone_frame = frame.copy() # Copy the frame
    frame_roi = crop2roi(frame,roi_pts,mask)
    mask2 = thresh(mask, roi_pts, frame)
    contours, contour_info, position_info = detectContours(frame_roi,mask2,contour_info,position_info,blackOnWhite)

    center = position_info[int(k)]
    cv2.putText(clone_frame,'Frame: %d' % k,(10,50), font,1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(clone_frame,'Time: %.2f' % time,(10,100), font,1,(255,255,255),2,cv2.LINE_AA)
    cv2.circle(clone_frame, (center[0], center[1]), 7, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('frame',clone_frame)
    if save_vid.startswith('y'):
        out.write(clone_frame)

    i = i+1 #increase frame count
    if i == frame_count:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break