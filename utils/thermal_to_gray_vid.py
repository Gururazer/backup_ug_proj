import cv2

# Function to convert from one colormap to another
def convert_colormap(frame, from_colormap, to_colormap):
    # Apply the "from" colormap
    from_color_frame = cv2.applyColorMap(frame, from_colormap)
    # Convert to gray scale
    gray_frame = cv2.cvtColor(from_color_frame, cv2.COLOR_BGR2GRAY)
    # Apply the "to" colormap
    to_color_frame = cv2.applyColorMap(gray_frame, to_colormap)
    return to_color_frame

# Open the video file
cap = cv2.VideoCapture('path/to/input_video.mp4')

# Get the width and height of the frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
out = cv2.VideoWriter('path/to/output_video.mp4', fourcc, 20.0, (width, height))

# Specify the "from" and "to" colormaps
from_colormap = cv2.COLORMAP_INFERNO
to_colormap = cv2.COLORMAP_BONE

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert frame
        converted_frame = convert_colormap(frame, from_colormap, to_colormap)
        
        # Write the frame to the output video
        out.write(converted_frame)

        # Display the resulting frame (optional)
        cv2.imshow('Frame', converted_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
