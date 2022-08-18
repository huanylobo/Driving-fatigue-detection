# Driving-fatigue-detection
A simple program that uses the HAAR cascade to recognize blinks and yawns, and uses the PERCLOS formula to determine fatigue.
## 1.requirements
Enter the following command on the command line to get the required toolkit:
    
    $ pip install -r requirements.txt
 Then you can download `one-file.py` and `shape_predictor_68_face_landmarks.dat` which are real usaful part in that file.
 ## 2.Details about programme
 First, we'll try to find facial landmarks using dlib, then we'll detect different parts of the face, such as the left eye, nose, chin, etc. We will then extend this application to continuous live video streaming. With that done, we'll try to find the number of blinks using the concept of `eye aspect ratio` or EAR.
 
 Generally, we choose `0.20` as the EAR threshold, and when the calculated EAR is greater than 0.25, we output eyes open on the image.
When we detect eyes closed in 20 out of 300 frames, we will output drowsiness detect and activate `alarm=True`. When `alarm==true`, the system will issue an alarm. (alarm section is included in one-file-with-alarm)

 
