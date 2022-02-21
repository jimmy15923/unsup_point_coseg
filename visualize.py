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

