import torch
from ultralytics import YOLO
import cv2


import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is enabled



import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return 1 or more
print(torch.cuda.get_device_name(0))  # Should show your GPU name






# Load YOLO model and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("best2.pt").to(device)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)





while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor and move to GPU
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = model(frame_rgb, imgsz=320, conf=0.4)  # Faster processing

    # #2 Run YOLOv9 detection on the frame
    # results = model(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pothole: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


#using video feed

# import cv2
# import torch
# from ultralytics import YOLO

# # Load the YOLOv9 model
# model = YOLO("best2.pt")  # Make sure 'best2.pt' is the correct trained model

# # Load a video file instead of real-time webcam feed
# video_path = "vdofeed.mp4"  # Change to your video file path
# cap = cv2.VideoCapture(video_path)

# # Check if the video is opened
# if not cap.isOpened():
#     print("Error: Couldn't open the video file.")
#     exit()

# while cap.isOpened():
#     ret, frame = cap.read()  # Read each frame from the video
#     if not ret:
#         break  # Break when video ends

#     # Run YOLOv9 detection on the frame
#     results = model(frame)

#     # Draw detections on the frame
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
#             conf = float(box.conf[0])  # Confidence score
#             cls = int(box.cls[0])  # Class ID

#             # Draw bounding box & label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f"{model.names[cls]}: {conf:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



#     # Show the processed frame
#     cv2.namedWindow("Video Output", cv2.WINDOW_NORMAL)  # Allows manual resizing
#     cv2.resizeWindow("Video Output", 800, 600)  # Set an initial size


#     cv2.namedWindow("Video Output", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Video Output", 800, 600)
#     cv2.imshow("Video Output", frame_resized)

#     cv2.imshow("YOLOv9 Video Detection", frame)

    

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import torch
# from ultralytics import YOLO

# # Load the YOLO model
# model = YOLO(r"D:\01 Final Project\yolov9\best2.pt")  # Ensure correct path

# # Load the video file
# video_path = r"vdofeed.mp4"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print(f"❌ Error: Couldn't open the video file at {video_path}")
# else:
#     print("✅ Video loaded successfully.")

#     cv2.namedWindow("Pothole Detection", cv2.WINDOW_NORMAL)  # Allows resizing
#     cv2.resizeWindow("Pothole Detection", 800, 600)  # Adjust window size

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ No more frames or error reading the video.")
#             break

#         # Run the YOLO model on the original frame
#         results = model(frame)

#         # Debug: Print detection results
#         print(results)  # Check if detections exist

#         # Draw detection results on the frame
#         annotated_frame = results[0].plot()  

#         # Resize only the displayed frame, not the input to YOLO
#         display_frame = cv2.resize(annotated_frame, (800, 600))

#         cv2.imshow("Pothole Detection", display_frame)

#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
