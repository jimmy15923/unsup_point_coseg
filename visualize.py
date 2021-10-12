import cv2
import glob
import imageio
import open3d as o3d
import numpy as np


def custom_save(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1024, height=1024)
    vis.add_geometry(pcd)
    print(vis)
    for i in range(240):
        ctr = vis.get_view_control() 
        ctr.rotate(30, -0.5)
        ctr.set_zoom(-2.5)
        ren = vis.get_render_option()
        ren.point_size = 5
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("/tmp/pointcloud_{0:04d}.png".format(i))
    vis.destroy_window()



print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud('./work_dirs/raw/s3dis_150x/chair/1245.pcd')
print(pcd)
custom_save(pcd)


imgs = glob.glob("/tmp/pointcloud_*.png")
arrs = []

for filename in imgs:
    arrs.append(imageio.imread(filename))
imageio.mimsave('test.gif', arrs)














# # data = scipy.io.loadmat('../vis_data/chair.mat')
# # data = scipy.io.loadmat('./work_dirs/raw/s3dis/chair/results_n20_0.49.mat')
# data = scipy.io.loadmat('./work_dirs/raw/s3dis/chair/test_results.mat')

# y_preds = data['y_preds']
# coords = data['coord']
# colors = data['color']

# idx = 50

# coord = coords[idx]
# y_pred = y_preds[idx]

# # colors = np.zeros(shape=(*y_pred.shape, 3))
# color = colors[idx]
# color[y_pred>0] = (1,0.58,0.54)

# object_pcd = o3d.geometry.PointCloud()
# object_pcd.points = o3d.utility.Vector3dVector(coord)
# object_pcd.colors = o3d.utility.Vector3dVector(color)

# # visualizer = JVisualizer()
# visualizer = o3d.visualization.Visualizer()
# visualizer.create_window()
# visualizer.add_geometry(object_pcd)
# # visualizer.show() 
# visualizer.capture_screen_image('./image_1.png')