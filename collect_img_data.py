import os
import cv2


# We are creating the directory for the data. data_directory defines a variable and assigns it to the string value './data'. Inteded to store the collected images.
data_directory = './data'
if not os.path.exists(data_directory): 
    os.makedirs(data_directory)

letters_in_alphabet = 26 # Number of letters in alphabet
size_of_dataset = 100

capture = cv2.VideoCapture(0) # Initializes a video capture object using OpenCV

for i in range(letters_in_alphabet):
    if not os.path.exists(os.path.join(data_directory, str(i))):
        os.makedirs(os.path.join(data_directory, str(i)))

    print('Collecting data for class {}'.format(i))

    while True: # statement to keep displaying video feed
        success, frame = capture.read()
        cv2.putText(frame, 'Ready? Press "Q"', (500,50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    count = 0 # Keep track of number of images collected for the current class 'i'.
    while count < size_of_dataset:
        success, frame = capture.read()
        cv2.imshow('frame', frame) # displays the captured frame in a window titled 'frame'.
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_directory, str(i), '{}.jpg'.format(count)), frame)

        count += 1

capture.release() # Releases the video capture object 'capture'.
cv2.destroyAllWindows() # Closes all OpenCV display windows.