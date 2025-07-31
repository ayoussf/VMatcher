# Code adapted from EfficientLoFTR [https://github.com/zju3dv/EfficientLoFTR/]

import torch
import cv2
import poselib
from .warppers import Camera, Pose
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous

# --- METRICS ---
def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

def estimate_homography(data, config):
    """
    Estimate homography between two sets of keypoints.
    Inputs:
        m_kpts0: (N, 2) array of (x, y) keypoint coordinates
        m_kpts1: (N, 2) array of (x, y) keypoint coordinates
        H: (3, 3) array
        shape: (height, width) of the image
        dist_thresh: threshold on matching distance
    Outputs:
        correctness: 1 if estimated homography is correct, 0 otherwise
        mean_dist: mean distance between true warped points and estimated warped points
        inliers: (N, ) array of inliers
    """
    data.update({'mean_dist': []})
    kpts0 = data['mkpts0_f']
    kpts1 = data['mkpts1_f']
    mconf = data['mconf']
    shape0, shape1 = data['HPatches_size']
    
    # Select Top-K matches
    topk = config.topk if config.topk > 0 else None
    if topk is not None:
        if len(kpts0) > topk:
            mconf, indicies = torch.topk(mconf, k=topk, dim=0, largest=True, sorted=True)
            m_kpts0 = kpts0[indicies].cpu().numpy()
            m_kpts1 = kpts1[indicies].cpu().numpy()
        elif len(kpts0) > 4 and len(kpts0) <= topk:
            m_kpts0 = kpts0.cpu().numpy()
            m_kpts1 = kpts1.cpu().numpy()
        else:
            return data['mean_dist'].append(float('inf'))
     
    if config.homography_estimation_method == 'CV':
        H = data['H'].squeeze().cpu().numpy()
        shape0, shape1 = shape0.item(), shape1.item()
        estimated_H, inliers = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, maxIters=10000, confidence=0.99999, ransacReprojThreshold=config.ransac_pixel_thr)
        if estimated_H is None:
            data['mean_dist'].append(float('inf'))
        else:
            inliers = inliers.flatten()
            corners = np.array([[0, 0, 1],
                                [shape1 - 1, 0, 1],
                                [0, shape0 - 1, 1],
                                [shape1 - 1, shape0 - 1, 1]])
            
            real_warped_corners = np.dot(corners, np.transpose(H))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(estimated_H))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            data['mean_dist'].append(mean_dist)
            data['inliers'] = inliers

    
    elif config.homography_estimation_method == 'Poselib':
        H = data['H'].squeeze()
        M, info = poselib.estimate_homography(m_kpts0,
                                              m_kpts1,
                                              {"max_reproj_error": config.ransac_pixel_thr,},)
        if M is None:
            data['mean_dist'].append(float('inf'))
        else:
            M = torch.tensor(M).to(kpts0)
            inl = torch.tensor(info["inliers"]).bool().to(kpts0.device)
            corners0 = torch.Tensor([[0, 0], [shape1, 0], [shape1, shape0], [0, shape0]]).float().to(M)
            corners1_gt = from_homogeneous(to_homogeneous(corners0) @ H.transpose(-1, -2))
            corners1 = from_homogeneous(to_homogeneous(corners0) @ M.transpose(-1, -2))
            d = torch.sqrt(((corners1 - corners1_gt) ** 2).sum(-1))
            mean_dist = d.mean(-1).cpu().numpy()
            data['mean_dist'].append(mean_dist)
            data['inliers'] = inl.cpu().numpy()

def hpatches_auc(errors: list) -> list:
    """
    Compute AUC scores for different thresholds.
    Inputs:
        errors: list of errors
        thresholds: list of thresholds
    Outputs:
        aucs: list of auc scores
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in [3, 5, 10]:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return {'auc@3': aucs[0], 'auc@5': aucs[1], 'auc@10': aucs[2]}

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def estimate_lo_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    camera0, camera1 = Camera.from_calibration_matrix(K0).float(), Camera.from_calibration_matrix(K1).float()
    pts0, pts1 = kpts0, kpts1

    M, info = poselib.estimate_relative_pose(
        pts0,
        pts1,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": thresh,
            "success_prob": conf,
        },
    )
    success = M is not None and ( ((M.t != [0., 0., 0.]).all()) or ((M.q != [1., 0., 0., 0.]).all()) )
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t))
    else:
        M = Pose.from_4x4mat(torch.eye(4).numpy())

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")),
        **info,
    }
    return estimation


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.ransac_pixel_thr  # 0.5
    conf = config.ransac_conf  # 0.99999
    RANSAC = config.pose_estimation_method
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        assert config.eval_times >= 1, "eval_times should be >= 1"
        if config.eval_times >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            R_list, T_list, inliers_list = [], [], []
            for _ in range(config.eval_times):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]
                
                if RANSAC == 'RANSAC':
                    ret = estimate_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if ret is None:
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                        R_list.append(R_err)
                        T_list.append(t_err)
                        inliers_list.append(inliers)

                elif RANSAC == 'LO-RANSAC':
                    est = estimate_lo_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if not est["success"]:
                        R_list.append(float("inf"))
                        T_list.append(float("inf"))
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        M = est["M_0to1"]
                        inl = est["inliers"].numpy()
                        t_error, r_error = relative_pose_error(T_0to1[bs], M.R, M.t, ignore_gt_t_thr=0.0)
                        R_list.append(r_error)
                        T_list.append(t_error)
                        inliers_list.append(inl)
                else:
                    raise ValueError(f"Unknown RANSAC method: {RANSAC}")

            data['R_errs'].append(R_list)
            data['t_errs'].append(T_list)
            data['inliers'].append(inliers_list[0])


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4, config=None, hpatches=False):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')
    if hpatches:
        aucs = hpatches_auc(metrics['mean_dist'])
        return {**aucs}
    else:
        # pose auc
        angular_thresholds = [5, 10, 20]

        if config.eval_times >= 1:
            pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, config.eval_times)[unq_ids].reshape(-1)
        else:
            pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
        aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

        # matching precision
        dist_thresholds = [epi_err_thr]
        precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
        
        u_num_mathces = np.array(metrics['num_matches'], dtype=object)[unq_ids]
        num_matches = {f'num_matches': u_num_mathces.mean() }
        return {**aucs, **precs, **num_matches}