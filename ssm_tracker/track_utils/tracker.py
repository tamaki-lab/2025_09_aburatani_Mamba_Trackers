"""
Naive Base Tracker, mostly borrow from bytetrack
"""

import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import MambaTracklet
from .matching import *

class MambaTracker(object):
    '''
    MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model
    '''
    def __init__(self, cfgs, device):
        self.tracked_tracklets = []  
        self.lost_tracklets = []  
        self.removed_tracklets = []  

        self.frame_id = 0
        self.cfgs = cfgs

        self.det_thresh = self.cfgs['filter_thresh']
        self.new_track_thresh = self.cfgs['new_track_thresh']
        self.max_time_lost = self.cfgs['max_time_lost']

        # set motion predictor
        self.device = device
        MambaTracklet.set_motion_predictor(motion=cfgs['model'],
                                           cfgs=self.cfgs, 
                                           device=device)


    def update(self, output_results, meta_data=None):
        """
        output_results: processed detections (scale to original size) tlwh format
        meta_data: List[float], img h and img w
        """

        self.frame_id += 1
        activated_starcks = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        categories = output_results[:, 5]
        bboxes = output_results[:, :4]

        # convert tlwh to xywh
        bboxes[:, :2] += 0.5 * bboxes[:, 2:]

        remain_inds = scores > self.cfgs['filter_thresh']
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cates_keep = categories[remain_inds]


        if len(dets) > 0:
            detections = [MambaTracklet(self.cfgs, xywh, s, categoty, img_h=meta_data[0], img_w=meta_data[1], device=self.device) 
                          for (xywh, s, categoty) in zip(dets, scores_keep, cates_keep)]
        else:
            detections = []

        # predict traj motions
        tracklet_pool = joint_tracklets(self.tracked_tracklets, self.lost_tracklets)
        for tracklet in tracklet_pool:
            tracklet.predict()

        # round one: associate detections with activate tracklets
        dists = iou_distance(self.tracked_tracklets, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.7)

        # for matched pairs, update the active tracklets
        for itracked, idet in matches:
            track = self.tracked_tracklets[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        # round two: associate unmtached detections with lost tracklets
        
        unmatched_detections = [detections[i] for i in u_detection]
        unmatched_tracklets = [self.tracked_tracklets[i] for i in u_track]
        unmatched_tracklets = joint_tracklets(unmatched_tracklets, self.lost_tracklets)

        dists = iou_distance(unmatched_tracklets, unmatched_detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = unmatched_tracklets[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)  # for matched lost tracklets, refine it

        for it in u_track:
            # update with last predicted bbox
            track = unmatched_tracklets[it]
            track.update(None, self.frame_id)
            track.mark_lost()
            lost_tracklets.append(track)


        # initial new detections
        for inew in u_detection:
            track = unmatched_detections[inew]
            if track.score < self.new_track_thresh:
                continue    
            track.activate(self.frame_id)
            activated_starcks.append(track)    

        # remove long lost tracklets
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)


        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, activated_starcks)
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)
        # get scores of lost tracks
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets


def joint_tracklets(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_tracklets(tlista, tlistb):
    tracklets = {}
    for t in tlista:
        tracklets[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tracklets.get(tid, 0):
            del tracklets[tid]
    return list(tracklets.values())


def remove_duplicate_tracklets(trackletsa, trackletsb):
    pdist = iou_distance(trackletsa, trackletsb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = trackletsa[p].frame_id - trackletsa[p].start_frame
        timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
    resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
    return resa, resb