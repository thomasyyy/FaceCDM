import torch
import torchvision.transforms as transforms
from PIL import Image
# from SD_style.insightface.face_align import FaceAlignment, LandmarksType
import face_alignment

# 加载图像
image_path = '/root/FreeDoM/SD_style/POPO-screenshot-20230725-200824.jpg'
image = Image.open(image_path)

# 创建FaceAlignment对象
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='dlib')

# 转换图像
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
image_tensor = transform(image)

# 提取特征点
with torch.no_grad():
    image1 = image_tensor.cpu().numpy()
    landmarks = fa.get_landmarks(image1)
print(landmarks)
# 绘制特征点
landmarks = landmarks[0]
for x, y in landmarks:
    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
