import cv2

gray8_img = cv2.imread("D:/FLIR_DATASET/FLIR_ADAS_v2/video_thermal_test/data/video-4FRnNpmSmwktFJKjg-frame-001030-wZYKJYhwC8XDtx3ka.jpg",cv2.IMREAD_ANYDEPTH)

inferno = cv2.applyColorMap(gray8_img,cv2.COLORMAP_INFERNO)
gray = cv2.applyColorMap(inferno,cv2.COLORMAP_BONE)
cv2.imshow("inf",inferno)
cv2.imshow("gr",gray)
cv2.waitKey(0)