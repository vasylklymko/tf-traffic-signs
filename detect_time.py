import cv2

def display_detect_time(image, time):
    fps_text = f"D_time: {time:.4f}"
    cv2.putText(image, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image