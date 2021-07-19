from abc import ABC, abstractmethod
import urllib
import threading
import numpy as np
import cv2


class AbstractFrame(ABC):
    @abstractmethod
    def get_frame(self):
        pass


class JpgServer(AbstractFrame):

    def __init__(self, url: str) -> None:
        self.frame = None
        self.url = url

    @property
    def get_frame(self):
        return self.frame


class MJPEGStream(AbstractFrame):
    def __init__(self, url: str) -> None:
        self.url = url
        self.stream = MJPEGStream(self.url)
        self.stream.start()
        self.frame = None
        self.thread = threading.Thread(target=self.start)
        self.thread.daemon = True
        self.thread.start()

    @property
    def get_frame(self):
        return self.frame

    def start(self):
        while True:
            self.frame = self.stream.read()
            if self.frame is None:
                continue
                # work with pil or without
            self.frame = np.fromstring(self.frame, dtype=np.uint8)
            self.frame = cv2.resize(cv2.imdecode(self.frame, 1), (640, 480))


# size!!!!!!!!!!!!!!!!!!!!!!!!
class MjpgServer(AbstractFrame):
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.stream = urllib.request.urlopen(self.stream_url)
        self.data = b''
        self.frame = None
        self.thread = threading.Thread(target=self.start)
        self.thread.daemon = True
        self.thread.start()

    @property
    def get_frame(self):
        # Error for timing!!!!!!!!!!!!!!!!!!!!!!!!!!
        while True:
            if self.frame is None:
                continue
            self.frame = np.fromstring(self.frame, dtype=np.uint8)
            self.frame = cv2.resize(cv2.imdecode(self.frame, 1), (640, 480))
            break
        return self.frame

    def start(self):
        while True:
            self.data += self.stream.read(5120)
            start_pos = self.data.find(b'\xff\xd8')
            end_pos = self.data.find(b'\xff\xd9')
            if start_pos != -1 and end_pos != -1:
                self.frame = self.data[start_pos:end_pos + 2]
                self.data = self.data[end_pos + 2:]
