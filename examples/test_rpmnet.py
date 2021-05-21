import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))

from learning3d.models import RPMNet, PPFNet
from learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss
from learning3d.data_utils import RegistrationData, ModelNet40Data

def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)
            
            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)
        
    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()

def plot_geometries(label, geometries):
    plt.rcParams['figure.figsize'] = [16, 16]
    ax = plt.axes(projection='3d')
    ax.axis("off")
    ax.view_init(45, 45)
 
    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=colors)
    
    plt.savefig(label + '.png')
    # plt.show()

def display_open3d(label, template, source, transformed_source):
	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	transformed_source_ = o3d.geometry.PointCloud()
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	transformed_source_.paint_uniform_color([0, 0, 1])
	plot_geometries(label, [template_, source_, transformed_source_])

def test_one_epoch(name, device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt = data

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		output = model(template, source)
		
		if i % 5 == 0:
			label = name + str(i)
			display_open3d(label, template.detach().cpu().numpy()[0,:,:3], source.detach().cpu().numpy()[0,:,:3], output['transformed_source'].detach().cpu().numpy()[0])
		loss_val = FrobeniusNormLoss()(output['est_T'], igt)

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader):
	test_loss = test_one_epoch(args.exp_name, args.device, model, test_loader)
	print('Validation Loss: %f'%(test_loss))
	
def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_rpmnet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PointNet
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=10, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_rpmnet/models/partial-trained.pth', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()

	testset = RegistrationData('RPMNet', ModelNet40Data(train=False, num_points=args.num_points, use_normals=True), partial_source=True, partial_template=False)
	test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)
	
	# Create RPMNet Model.
	model = RPMNet(feature_model=PPFNet())
	model = model.to(args.device)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu')['state_dict'])
	model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()