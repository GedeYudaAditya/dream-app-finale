imageHeight = 500
imageWidth = 888
minDistance = imageHeight/6.25


class StreamText:
    def __init__(self, status, mean_fps, model=None):
        self.status = status
        self.mean_fps = mean_fps
        self.model = model


class UploadModel:
    def __init__(self, model_name, date, time):
        self.model_name = model_name
        self.date = date
        self.time = time
