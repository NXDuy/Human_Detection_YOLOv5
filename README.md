# Human_Detection_YOLOv5
Prolem: Counting the number of humans in webcam video, help teacher keep track of their student when they take an exams
In this project, I used YOLOv5m to train my model. 
- YOLOv5s is also good and small model. But it accuracy is not good in this problem.
- However, YOLOv5m is bigger than YOLOv5 but not too big to slow down the computation very much
- With Nvidia 1650Ti 4GB my model achieve is 30 FPS
- Accuracy achieved: 85%
- Training and testing data: COCO data
- Database is used for saving data: SQLite

However, this project is only good for counting people in the video. Not good for the temporal segment in the video, in each segment, displays the number of human the change of it in this segment

In later version of project, I recognize the background around student is not changed too much, so LSTM will be used in next verision to achieve better accuracy and temporal segment in video

To run this project:

```python3 dash_display.py```
