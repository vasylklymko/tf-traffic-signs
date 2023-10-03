import cv2

def display_detect_time(image, time, prompt_text = "D_time"):
    fps_text = f"{prompt_text}: {time:.1f}"
    cv2.putText(image, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image