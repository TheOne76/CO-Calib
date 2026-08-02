import sm

import numpy as np
import sys
import multiprocessing
import os
try:
   import queue
except ImportError:
   import Queue as queue # python 2.x
import time
import copy
import cv2

# ---------------------------------------------------------------------------
# NN-Detector pre-computed detection injection
# ---------------------------------------------------------------------------

def _load_precomputed_detections(npz_path, topic):
    """
    Load corner detections produced by extract_fisheye_detections.py.

    Returns dict with keys 'timestamps_ns', 'xy_px', 'conf', or None on error.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError("precomputed detections not found: {}".format(npz_path))

    data = np.load(npz_path)
    key = topic.replace("/", "_").strip("_")

    ts_key   = "{}__timestamps_ns".format(key)
    xy_key   = "{}__xy_px".format(key)
    conf_key = "{}__conf".format(key)

    if ts_key not in data:
        raise KeyError(
            "Topic '{}' not found in {}.\n"
            "Available keys: {}".format(topic, npz_path, list(data.keys()))
        )

    return {
        "timestamps_ns": data[ts_key],  # (N,) int64
        "xy_px": data[xy_key],          # (N,36,4,2) float32
        "conf": data[conf_key],         # (N,36,4) float32
    }


def _timestamp_to_ns(stamp):
    """Convert aslam::Time / rospy.Time / sm.Time to integer nanoseconds."""
    for attr in ("to_nsec", "toNSec", "getNSec"):
        fn = getattr(stamp, attr, None)
        if fn is not None:
            return int(fn())
    return int(stamp.toSec() * 1e9)


def _find_nearest_frame(timestamps_ns, query_ns, tol_ns=5_000_000):
    """
    Find index of closest timestamp within tol_ns (default 5 ms).
    Returns -1 if no match found.
    """
    diff = np.abs(timestamps_ns.astype(np.int64) - np.int64(query_ns))
    idx = int(np.argmin(diff))
    if diff[idx] <= tol_ns:
        return idx
    return -1


def _build_observation_from_detections(detector, timestamp, image,
                                        xy_px, conf,
                                        conf_thr=0.99, min_corners=4):
    """
    Construct a GridCalibrationTargetObservation populated with
    NN-Detector corners, bypassing the C++ ethz_apriltag2 detector.

    Note:
      The precomputed NN-Detector layout consumed successfully by kalibr in
      this repository is:
        index 0 = TopRight
        index 1 = TopLeft
        index 2 = BottomLeft
        index 3 = BottomRight
    """
    try:
        import aslam_cv as acv
    except ImportError:
        import aslam_cv_python as acv

    target = detector.target()
    tag_rows = target.rows() // 2
    tag_cols = target.cols() // 2
    grid_cols = target.cols()

    obs = acv.GridCalibrationTargetObservation(target)
    obs.setTime(timestamp)
    obs.setImage(np.array(image))

    # NN-Detector corner slot -> (target_row_offset, target_col_offset)
    _FC_TO_DR_DC = [
        (1, 1),  # 0=TopRight
        (1, 0),  # 1=TopLeft
        (0, 0),  # 2=BottomLeft
        (0, 1),  # 3=BottomRight
    ]

    n_injected = 0
    for tag_r in range(tag_rows):
        for tag_c in range(tag_cols):
            tag_idx = tag_r * tag_cols + tag_c
            for fish_ci, (dr, dc) in enumerate(_FC_TO_DR_DC):
                if conf[tag_idx, fish_ci] < conf_thr:
                    continue
                flat_idx = (2 * tag_r + dr) * grid_cols + (2 * tag_c + dc)
                xy = xy_px[tag_idx, fish_ci].astype(np.float64)
                obs.updateImagePoint(flat_idx, xy)
                n_injected += 1

    success = n_injected >= min_corners
    return success, obs


# ---------------------------------------------------------------------------
# Original extraction logic (with optional NN-Detector injection)
# ---------------------------------------------------------------------------

def multicoreExtractionWrapper(detector, taskq, resultq, clearImages, noTransformation):
    while 1:
        try:
            task = taskq.get_nowait()
        except queue.Empty:
            return
        idx = task[0]
        stamp = task[1]
        image = task[2]

        if noTransformation:
            success, obs = detector.findTargetNoTransformation(stamp, np.array(image))
        else:
            success, obs = detector.findTarget(stamp, np.array(image))

        if clearImages:
            obs.clearImage()
        if success:
            resultq.put((obs, idx))


def extractCornersFromDataset(dataset, detector, multithreading=False,
                               numProcesses=None, clearImages=True,
                               noTransformation=False,
                               precomputed_detections_file=None,
                               conf_thr=0.99,
                               min_corners=4):
    """
    Extract calibration target corners from a dataset.

    If precomputed_detections_file is given (path to .npz produced by
    omnicalib-cache-export), NN-Detector results are injected
    directly into the observation objects, bypassing the C++ AprilTag detector.
    """
    print("Extracting calibration target corners")
    targetObservations = []
    numImages = dataset.numImages()

    iProgress = sm.Progress2(numImages)
    iProgress.sample()

    if precomputed_detections_file is not None:
        topic = dataset.topic
        print("  [NN-Detector] Loading precomputed detections for topic: {}".format(topic))
        det_data = _load_precomputed_detections(precomputed_detections_file, topic)
        ts_arr = det_data["timestamps_ns"]
        xy_arr = det_data["xy_px"]
        conf_arr = det_data["conf"]
        print("  [NN-Detector] Loaded {} frames, conf_thr={}, min_corners={}".format(
              len(ts_arr), conf_thr, min_corners))

        matched = 0
        for timestamp, image in dataset.readDataset():
            query_ns = _timestamp_to_ns(timestamp)
            frame_idx = _find_nearest_frame(ts_arr, query_ns)

            if frame_idx < 0:
                iProgress.sample()
                continue

            xy_px = xy_arr[frame_idx]
            conf = conf_arr[frame_idx]

            success, observation = _build_observation_from_detections(
                detector,
                timestamp,
                image,
                xy_px,
                conf,
                conf_thr=conf_thr,
                min_corners=min_corners,
            )

            if clearImages:
                observation.clearImage()
            if success:
                targetObservations.append(observation)
                matched += 1
            iProgress.sample()

        print("\r  [NN-Detector] Accepted {}/{} frames".format(matched, numImages))

    elif multithreading:
        if not numProcesses:
            numProcesses = max(1, multiprocessing.cpu_count() - 1)
        try:
            manager = multiprocessing.Manager()
            resultq = manager.Queue()
            manager2 = multiprocessing.Manager()
            taskq = manager2.Queue()

            for idx, (timestamp, image) in enumerate(dataset.readDataset()):
                taskq.put((idx, timestamp, image))

            plist = list()
            for pidx in range(0, numProcesses):
                detector_copy = copy.copy(detector)
                p = multiprocessing.Process(
                    target=multicoreExtractionWrapper,
                    args=(detector_copy, taskq, resultq, clearImages, noTransformation))
                p.start()
                plist.append(p)

            last_done = 0
            while 1:
                if all([not p.is_alive() for p in plist]):
                    time.sleep(0.1)
                    break
                done = numImages - taskq.qsize()
                sys.stdout.flush()
                if (done - last_done) > 0:
                    iProgress.sample(done - last_done)
                last_done = done
                time.sleep(0.5)
            resultq.put('STOP')
        except Exception as e:
            raise RuntimeError("Exception during multithreaded extraction: {0}".format(e))

        if resultq.qsize() > 1:
            targetObservations = [[]] * (resultq.qsize() - 1)
            for lidx, data in enumerate(iter(resultq.get, 'STOP')):
                obs = data[0]
                time_idx = data[1]
                targetObservations[lidx] = (time_idx, obs)
            targetObservations = list(zip(*sorted(targetObservations, key=lambda tup: tup[0])))[1]
        else:
            targetObservations = []

    else:
        for timestamp, image in dataset.readDataset():
            if noTransformation:
                success, observation = detector.findTargetNoTransformation(timestamp, np.array(image))
            else:
                success, observation = detector.findTarget(timestamp, np.array(image))
            if clearImages:
                observation.clearImage()
            if success == 1:
                targetObservations.append(observation)
            iProgress.sample()

    if len(targetObservations) == 0:
        print("\r")
        sm.logFatal("No corners could be extracted for camera {0}! Check the calibration target configuration and dataset.".format(dataset.topic))
    else:
        print("\r  Extracted corners for %d images (of %d images)                              " % (len(targetObservations), numImages))

    cv2.destroyAllWindows()
    return targetObservations
