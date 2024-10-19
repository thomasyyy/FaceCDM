import cv2
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import transform as trans
import torchvision.transforms as transforms
import torchvision
from mtcnn import MTCNN
from einops import rearrange
import torch.nn.functional as F
from skimage import transform as trans
class FaceAligner:
    def __init__(self, image_size=256) -> None:
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)
        self.src = src * image_size / 112
        self.tform = trans.SimilarityTransform()
        self.A = None
        self.B = None

    def get_trans_matrix(self, w1, h1):
        A = [[2 / (w1 - 1), 0, -1],
             [0, 2 / (h1 - 1), -1],
             [0, 0, 1]]
        B = [[2 / 255, 0, -1],
             [0, 2 / 255, -1],
             [0, 0, 1]]
        self.A = np.array(A).astype(np.float32)
        self.B = np.array(B).astype(np.float32)

    def align(self, lmks, process_src=True):
        self.tform.estimate(lmks, self.src)
        if process_src:
            return torch.from_numpy(self.tform.params[0:2, :]).unsqueeze(0).to(torch.device("cuda")), None
        image_to_face_mat = self.tform.params
        theta = self.A.dot(np.linalg.inv(image_to_face_mat)).dot(np.linalg.inv(self.B)).astype(np.float32)
        theta_inverse = self.B.dot(image_to_face_mat).dot(np.linalg.inv(self.A)).astype(np.float32)
        theta = torch.from_numpy(theta[0:2]).unsqueeze(0).to(torch.device("cuda"))
        theta_inverse = torch.from_numpy(theta_inverse[0:2]).unsqueeze(0).to(torch.device("cuda"))
        return theta, theta_inverse
img = "/root/FreeDoM/SD_style/intermediates.jpg"
# img = cv2.imread(img)
ref = Image.open(img)
detector = MTCNN()
face_aligner = FaceAligner()
device = torch.device('cuda:0')
ouput_size = 112
ori_img = cv2.cvtColor(np.asarray(ref), cv2.COLOR_BGR2RGB)
r = detector.detect_faces(ori_img)
if face_aligner.A is None:
    face_aligner.get_trans_matrix(ori_img.shape[1], ori_img.shape[0])
target_lmks = np.array(list(r[0]['keypoints'].values()))
print(target_lmks)
def draw_on(img, target_lmks):
    import cv2
    dimg = img.copy()
    kps = target_lmks.astype(int)
    #print(landmark.shape)
    for l in range(len(target_lmks)):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                    2)
    cv2.imwrite("./t1_output.jpg", dimg)
img_draw = draw_on(ori_img, target_lmks)
theta, theta_inverse = face_aligner.align(target_lmks, process_src=False)
print("theta",theta)
ori_image = torch.from_numpy(ori_img.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).contiguous().to(device)
print("ori_image",ori_image)
grid = F.affine_grid(theta, [ori_image.shape[0], 3, ouput_size, ouput_size],
                        align_corners=True)
print("grid",grid)
detected_face = F.grid_sample(ori_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
print("detected_face",detected_face)
# detected_face = detected_face.div(255.0)
# detected_face_ref_tensor = (detected_face - 0.5) / 0.5
# detected_face_norm = ((detected_face - 0.5) / 0.5) # 归一化处理
detected_face_norm = detected_face.squeeze(0) # 将像素值限制在0-1范围内
detected_face_norm = detected_face_norm.permute(1, 2, 0).cpu().numpy() # 将通道维度调整到最后一个维度
cv2.imwrite('detected_face.jpg',detected_face_norm)
# detected_face = detected_face.div(255.0)
# detected_face = (detected_face - 0.5) / 0.5
# face_img = crop_img(faces[0]['bbox'].tolist(), img)
# print(face_img.shape)
# aimg = face_align.norm_crop(img, landmark=faces[0].kps, image_size=256)
# aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
# # 进行人脸放射变换
# detected_face_norm = torch.clamp(detected_face, 0, 1).squeeze(0) # 将像素值限制在0-1范围内
# print(detected_face_norm.shape)
# detected_face_norm = detected_face_norm.permute(1, 2, 0) # 将通道维度调整到最后一个维度
# img = Image.fromarray((detected_face_norm.cpu() * 255).numpy().astype('uint8'))

# # 保存图片
# img.save('detected_face.jpg')
# tensor_img = torchvision.transforms.functional.affine(tensor_img, angle=angle.item(), translate=[0, 0], scale=1.0, shear=0.0)
# transform = transforms.Compose([
#     # transforms.Resize((512, 512)),
#     # transforms.CenterCrop((448, 448)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: torchvision.transforms.functional.affine(x, angle=angle.item(), translate=[0, 0], scale=1.0, shear=0.0)),
#     # transforms.Lambda(lambda x: F.crop(x, (32, 32, 416, 416))),
# ])
# tensor_img = transform(face_img)
# transform1 = transforms.Compose([
#         transforms.ToPILImage(),
#     ])
# print(tensor_img.mean())
# pil_img = transform1(tensor_img)

# # pil_img.save('test.jpg')
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

# handler = insightface.model_zoo.get_model('your_recognition_model.onnx')
# handler.prepare(ctx_id=0)
# img = "/root/FreeDoM/SD_style/CqgNOlgkO6CAShCRAAAAAAAAAAA154.1180x842.jpg"
# img = cv2.imread(img)
# faces = handler.get(img)
# print(faces)
# import argparse
# import cv2
# import sys
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

# assert insightface.__version__>='0.3'

# parser = argparse.ArgumentParser(description='insightface app test')
# # general
# parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
# parser.add_argument('--det-size', default=640, type=int, help='detection size')
# args = parser.parse_args()

# app = FaceAnalysis()
# app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

# img = "/root/FreeDoM/SD_style/POPO-screenshot-20230725-200824.jpg"
# img = cv2.imread(img)
# faces = app.get(img)
# print(faces[0].kps.astype(int))
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# # then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)

