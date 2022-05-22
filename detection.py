import cv2
import torch
from time import time
import numpy as np

class HumanDetection:
    def __init__(self, videoLink):
        self.video = cv2.VideoCapture(videoLink)
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('\nDevice is used', self.device)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='.\\Model\\best.pt', force_reload=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord= results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x_class):
        return self.classes[int(x_class)]
    
    def plot_boxes(self, results, fps, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape  = frame.shape[1], frame.shape[0]

        fps_color = (0, 255, 255)
        fps_position = (int(x_shape*0.75), int(y_shape*0.1))

        cv2.putText(frame, ('FPS: %.1f' % (fps,)), fps_position,
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, fps_color, 2)

        for i in range(n):
            row = cord[i]
            if row[4] > 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                color = (255,0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, ('%s %.2f'%(self.class_to_label(labels[i]), row[4])) , (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def __call__(self):
        player = self.video
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = 0.0
        while True:
            start_time = time()
            ret, frame = self.video.read()
            if not ret:
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, fps, frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)

            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

#videoDetect = HumanDetection(0)
#videoDetect()


        
