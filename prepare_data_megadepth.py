import sys
sys.path.append('../../submodules/glue_factory')

import argparse
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

# TODO: Add imports and configs for the Megadepth dataset and the followed detector (SuperPoint)
# TODO: Check what kind of ratio the NG-RANSAC paper or Reinforced Local Feature paper uses for SP

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Train large scale camera localization.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--variant', '-v', default='train', choices=['train', 'test'],
	help='Defines subfolders of the dataset ot use (split according to "Deep Fundamental Matrix", Ranftl and Koltun, ECCV 2018).')

parser.add_argument('--nfeatures', '-nf', type=int, default=-1, 
	help='number of features per image, -1 does not restrict feature count')

parser.add_argument('--conf_thresh', '-ct', type=int, default=0.00015, 
	help='confidence threshold')

parser.add_argument('--nms_dist', '-nd', type=int, default=4, 
	help='non maximum suppression distance')

parser.add_argument('--dataconf', '-dc', type=str, default='../../configs/superpoint+nearestneighbor_megadepth_dataconf_b1_grey.yaml',
	help='data configuration file path')

parser.add_argument('--modelparam', '-mp', type=str, default='../../pretrained/superpoint_v1_ours.pth',
	help='detector model parameter file path')

opt = parser.parse_args()

dataconf = OmegaConf.load(opt.dataconf)
dataset = get_dataset(dataconf.name)(dataconf)

if opt.variant == 'train':
	dataloader = dataset.get_data_loader("train", distributed=False, shuffle=True)
else:
	dataloader = dataset.get_data_loader("val", shuffle=True)
print('Using dataset: ', opt.dataconf, opt.variant)

# output folder that stores pre-calculated correspondence vectors as PyTorch tensors
out_dir = 'traindata/megadepth/' + opt.variant + '_data_sp/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

# setup detector
modelconf = {
	'name': 'two_view_pipeline',
	'extractor':
		{'name': 'gluefactory_nonfree.superpoint',
		'max_num_keypoints': opt.nfeatures,
		'force_num_keypoints': False,
		'detection_threshold': opt.conf_thresh,
		'nms_radius': opt.nms_dist,
		'trainable': False},
	'matcher':
		{'name': 'matchers.nearest_neighbor_matcher'},
	'ground_truth':
		{'name': 'matchers.depth_matcher',
		'th_positive': 3,
		'th_negative': 5,
		'th_epi': 5},
	'allow_no_extract': True
}
modelconf = OmegaConf.create(modelconf)
model = get_model(modelconf.name)(modelconf).to('cuda:0')
# model = get_model(modelconf.name)(modelconf)
model.load_state_dict(torch.load(opt.modelparam), strict=False)
model.eval()

with torch.no_grad():
	for i, data in enumerate(tqdm(dataloader)):
		img1_name = data['view0']['name'][0].split('.')[0]
		img2_name = data['view1']['name'][0].split('.')[0]

		tqdm.write("\nProcessing pair %d of %d. (%s, %s)" % (i, len(dataloader), img1_name, img2_name))

		pred = model(batch_to_device(data, 'cuda:0', non_blocking=True))
		# pred = model(data)

		img1_shape = data['view0']['image_size'][0].numpy() # W, H
		img1_shape = np.array([img1_shape[1], img1_shape[0], 3])
		img2_shape = data['view1']['image_size'][0].numpy() # W, H
		img2_shape = np.array([img2_shape[1], img2_shape[0], 3])

		K1 = data['view0']['camera'].calibration_matrix()[0] # B * 3 * 3
		K2 = data['view1']['camera'].calibration_matrix()[0] # B * 3 * 3

		GT_R_Rel = data['T_0to1'].R[0]
		GT_t_Rel = data['T_0to1'].t[0].unsqueeze(-1)

		matches1 = pred['matches0'][0]
		mask1 = matches1 != -1
		pts1 = pred['keypoints0'][0][mask1].unsqueeze(dim=0) # 1 * N * 2 x[x.nonzero(as_tuple=True)]
		pts2 = pred['keypoints1'][0][matches1[mask1]].unsqueeze(dim=0) # 1 * N * 2
		idx = torch.cat([torch.arange(end=mask1.clone().detach().int().sum(), device=matches1.device).unsqueeze(-1),
						torch.tensor(mask1[mask1.nonzero(as_tuple=True)], device=matches1.device).unsqueeze(-1)], dim=-1)
		ratios = pred['similarity'][0][idx[:,0],idx[:,1]].unsqueeze(dim=0).unsqueeze(dim=-1) # N

		# import matplotlib.pyplot as plt
		# for b in range(1):
		# 	f, axarr = plt.subplots(1,2, figsize=(20,10))
		# 	print("Shape of image: ", data['view0']['image'][b].shape, " and ", data['view1']['image'][b].shape, flush=True)
		# 	axarr[0].imshow((data['view0']['image'][b][0,:img1_shape[0], :img1_shape[1]]/255.).cpu().numpy())
		# 	axarr[1].imshow((data['view1']['image'][b][0,:img2_shape[0], :img2_shape[1]]/255.).cpu().numpy())
		# 	# Set title
		# 	plt.suptitle(f"sized {data['view0']['image_size'][b]} and {data['view1']['image_size'][b]}")
		# 	plt.savefig(f"./pair_{img1_name}_{img2_name}.png")
		# 	plt.clf()

		print("pts1: ", pts1.shape, pts1.device)
		print("pts2: ", pts2.shape, pts2.device)
		print("ratios: ", ratios.shape, ratios.device)
		print("img1: ", img1_shape)
		print("img2: ", img2_shape)
		print("K1: ", K1.shape, K1.device)
		print("K2: ", K2.shape, K2.device)
		print("GT_R_Rel: ", GT_R_Rel.shape, GT_R_Rel.device)
		print("GT_t_Rel: ", GT_t_Rel.shape, GT_t_Rel.device)
		exit()

		# Change the above code to pickle
		with open(out_dir + f'pair_{img1_name}_{img2_name}.pkl', 'wb') as f:
			pickle.dump([
				pts1.cpu().numpy().astype(np.float32), 
				pts2.cpu().numpy().astype(np.float32), 
				ratios.cpu().numpy().astype(np.float32), 
				img1_shape, 
				img2_shape, 
				K1.numpy().astype(np.float32), 
				K2.numpy().astype(np.float32), 
				GT_R_Rel.numpy().astype(np.float32), 
				GT_t_Rel.numpy().astype(np.float32)
				], f)

# pts1:  (1, N, 2)
# pts2:  (1, N, 2)
# ratios:  (1, N, 1)
# img1:  (H, W, 3)
# img2:  (H, W, 3)
# K1:  (3, 3)
# K2:  (3, 3)
# GT_R_Rel:  (3, 3)
# GT_t_Rel:  (3, 1)