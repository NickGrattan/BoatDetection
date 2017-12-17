#thread management
from queue import Queue
from collections import deque
import time

from Collector import Collector
from Detector import Detector
from Processor import Processor

# parameters
maxqueuelen=200

if __name__ == "__main__":
    # queue for images from camera
    queue_raw = Queue(maxsize=maxqueuelen)
    # queue for boat candidates & frame
    queue_detectors = Queue(maxsize=maxqueuelen)

    collector = Collector(queue_raw)
    detector = Detector(queue_raw, queue_detectors)
    processor = Processor(queue_detectors)
    
    collector.start()
    detector.start()
    processor.start()

    while True:
        print("queue_raw len: {} queue_detectors len: {}".\
            format( queue_raw.qsize(), queue_detectors.qsize()))
        time.sleep(60)

