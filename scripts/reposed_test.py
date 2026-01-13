import h5py
import numpy as np
import os

H5_OUTPUT = "scene_0015_matches.h5"

# Using a standard '0015' prefix and double underscore separator
# Many eval scripts look for this specific pattern to split the pair
img0 = "0015/dense0/imgs/10433230626_48a43692e0_o.jpg"
img1 = "0015/dense0/imgs/10537060136_357d6928e4_o.jpg"
pair_name = f"{img0.replace('/', '_')}__{img1.replace('/', '_')}"

with h5py.File(H5_OUTPUT, 'w') as f:
    grp = f.create_group(pair_name)
    n = 100 
    
    # Coordinates, Depth, Intrinsics, and Pose
    grp.create_dataset('mkpts0', data=np.random.rand(n, 2) * 1000)
    grp.create_dataset('mkpts1', data=np.random.rand(n, 2) * 1000)
    grp.create_dataset('depth0', data=np.random.rand(n))
    grp.create_dataset('depth1', data=np.random.rand(n))
    
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    grp.create_dataset('K0', data=K)
    grp.create_dataset('K1', data=K)
    grp.create_dataset('R', data=np.eye(3))
    grp.create_dataset('t', data=np.array([1.0, 0, 0]))

print(f"H5 Created with key: {pair_name}")