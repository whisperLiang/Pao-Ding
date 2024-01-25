import cv2
from torchvision.transforms import transforms


def get_ipt_from_video(capture: cv2.VideoCapture):
    ret, frame_bgr = capture.read()
    assert ret is True, "failed to read video"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((270, 480)),
        transforms.ToTensor()
    ])
    input_batch = preprocess(frame_rgb)
    return input_batch.unsqueeze(0)
