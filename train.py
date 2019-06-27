'''
Function:
	train the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import time
import torch
import config
import torchvision
import torch.nn as nn
from dataset import ImageFolder


'''print info'''
def Logging(message, savefile=None):
	content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
	if savefile:
		f = open(savefile, 'a')
		f.write(content + '\n')
		f.close()
	print(content)


'''train the model'''
def train():
	if not os.path.exists(config.backupdir):
		os.mkdir(config.backupdir)
	model = torchvision.models.resnet18(pretrained=True)
	model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
	use_cuda = torch.cuda.is_available() and config.use_cuda
	if use_cuda:
		os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
		if config.ngpus > 1:
			model = nn.DataParallel(model).cuda()
		else:
			model = model.cuda()
	model.train()
	dataloader = torch.utils.data.DataLoader(ImageFolder(config.imagespath,
														 config.labpath,
														 config.img_shape,
														 config.is_shuffle,
														 'train'),
											 batch_size=config.batch_size,
											 shuffle=False,
											 num_workers=config.num_workers)
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
	criterion = nn.MSELoss()
	for epoch in range(1, config.num_epochs+1):
		Logging('[INFO]: epoch now is %d...' % epoch, config.logfile)
		for batch_i, (_, imgs, targets) in enumerate(dataloader):
			imgs = imgs.type(FloatTensor)
			targets = targets.type(FloatTensor)
			optimizer.zero_grad()
			preds = model(imgs)
			loss = criterion(preds, targets)
			if config.ngpus > 1:
				loss = loss.sum()
			Logging('[INFO]: batch%d of epoch%d, loss is %.2f...' % (batch_i, epoch, loss.item()), config.logfile)
			loss.backward()
			optimizer.step()
		if (epoch % config.save_interval == 0) and (epoch > 0):
			pklpath = os.path.join(config.backupdir, 'epoch_%s.pkl' % str(epoch))
			if config.ngpus > 1:
				cur_model = model.module
			else:
				cur_model = model
			torch.save(cur_model.state_dict(), pklpath)
			acc = test(model)
			Logging('[INFO]: Accuracy of epoch %d is %.2f...' % (epoch, acc), config.logfile)


'''test the model'''
def test(model):
	model.eval()
	dataloader = torch.utils.data.DataLoader(ImageFolder(config.imagespath,
														 config.labpath,
														 config.img_shape,
														 config.is_shuffle,
														 'test'),
											 batch_size=config.batch_size,
											 shuffle=False,
											 num_workers=config.num_workers)
	use_cuda = torch.cuda.is_available() and config.use_cuda
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	n_correct = 0
	n_total = 0
	for batch_i, (_, imgs, targets) in enumerate(dataloader):
		imgs = imgs.type(FloatTensor)
		targets = targets.type(FloatTensor)
		preds = model(imgs)
		n_correct += (abs(targets - preds) < config.error_tolerance).sum().item()
		n_total += imgs.size(0)
	acc = n_correct / n_total
	model.train()
	return acc


if __name__ == '__main__':
	train()