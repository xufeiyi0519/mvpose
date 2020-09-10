"""Save the poses of different people to a new dictionary based on personid"""
from src.tools.getkey import get_key

def withID_dist (ID,pose):
    n = len(pose)
    pose_new = {}
    for i in range(n):
        k = get_key(ID,pose[i][0][0])
        pose_new[k] = pose[i]
    return pose_new