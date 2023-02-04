import copy
import os

import cv2
import numpy as np


class BaseCMC:
    def __init__(self, method='ecc', downscale=2, use_cmc_file=False, cmc_result_dir: str=None):
        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)

        else:
            self.method = None

        if use_cmc_file and cmc_result_dir is not None:
            try:
                self.cmc_files = os.listdir(cmc_result_dir)
                self.cmc_result_dir = cmc_result_dir
                self.tmp_cmc_file = None
                self.use_cmc_file = True
            except:
                self.use_cmc_file = False
        else:
            self.use_cmc_file = False

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame, vid_name=None, img_idx: int=None):
        if not self.use_cmc_file:
            if self.method == 'ecc':
                return self.applyEcc(raw_frame)
            elif self.method == 'sparseOptFlow':
                return self.applySparseOptFlow(raw_frame)
            elif self.method is None:
                return np.eye(2, 3)
            else:
                return np.eye(2, 3)
        else:
            return self.applyFile(vid_name, img_idx)

    def applyEcc(self, raw_frame: np.ndarray):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')
        return H

    def applySparseOptFlow(self, raw_frame: np.ndarray):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inlines = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale

        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def applyFile(self, vid_name, img_idx):
        if self.tmp_cmc_file is None:
            tmp_vid_name = vid_name if 'MOT17' not in vid_name else '-'.join(vid_name.split('-')[:-1])
            tmp_cmc_file = [x for x in self.cmc_files if tmp_vid_name in x][0]
            with open(os.path.join(self.cmc_result_dir, tmp_cmc_file)) as f:
                self.tmp_cmc_file = {int(x.split('\t')[0]): x[:-2].split('\t')[1:] for x in f.readlines()}

        try:
            tmp_cmc = self.tmp_cmc_file[img_idx]
            H = np.eye(2, 3, dtype=np.float_)
            H[0, 0] = float(tmp_cmc[0])
            H[0, 1] = float(tmp_cmc[1])
            H[0, 2] = float(tmp_cmc[2])
            H[1, 0] = float(tmp_cmc[3])
            H[1, 1] = float(tmp_cmc[4])
            H[1, 2] = float(tmp_cmc[5])
        except KeyError:
            H = np.eye(2, 3, dtype=np.float_)
        return H
