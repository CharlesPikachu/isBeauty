'''
Function:
	Predict.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize


'''predict'''
def predict(image_path, model_path):
	use_cuda = torch.cuda.is_available()
	model = torchvision.models.resnet18()
	model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
	model.load_state_dict(torch.load(model_path))
	if use_cuda:
		model = model.cuda()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	detector = dlib.get_frontal_face_detector()
	image = cv2.imread(image_path)
	b, g, r = cv2.split(image)
	image_rgb = cv2.merge([r, g, b])
	rects = detector(image_rgb, 1)
	if len(rects) >= 1:
		for rect in rects:
			lefttop_x = rect.left()
			lefttop_y = rect.top()
			rightbottom_x = rect.right()
			rightbottom_y = rect.bottom()
			cv2.rectangle(image, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)
			face = image_rgb[lefttop_y: rightbottom_y, lefttop_x: rightbottom_x] / 255.
			face = resize(face, (224, 224, 3), mode='reflect')
			face = np.transpose(face, (2, 0, 1))
			face = torch.from_numpy(face).float().resize_(1, 3, 224, 224)
			face = face.type(FloatTensor)
			res = round(model(face).item(), 2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, 'Value:'+str(res), (lefttop_x-5, lefttop_y-5), font, 0.5, (0, 0, 255), 1)
		cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('result', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


'''run'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Facial beauty predictor.")
	parser.add_argument('-i', dest='image', help='Image to be predicted.')
	parser.add_argument('-m', dest='model', help='The model path of facial beauty predictor.')
	args = parser.parse_args()
	if args.image and args.model:
		predict(args.image, args.model)