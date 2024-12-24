import numpy as np 
from .basetrack import BaseTrack, TrackState 
import torch
from copy import deepcopy

from loguru import logger

from models.MambaTrack import MambaTrack
from models.TrackSSM import TrackSSM

MOTION_PREDICTOR = {
    'MambaTrack': MambaTrack, 
    'TrackSSM': TrackSSM

}

def _preprocess_diff(bboxes):
    '''
    calculate the offset difference between frames, bbox format: xywh
    '''
    num_histories = bboxes.shape[0]
    assert num_histories > 1

    ret = np.zeros((num_histories - 1, 4))
    for i in range(num_histories - 1):
        ret[i] = bboxes[i + 1] - bboxes[i]

    return ret



class MambaTracklet(BaseTrack):
    motion_predictor = None  # share the motion predictor for saving the resource 

    @staticmethod
    def set_motion_predictor(motion, cfgs, device):
        logger.info(f'Initalizing motion predictor {motion}')
        model = MOTION_PREDICTOR[motion](cfgs).to(device)
        model.eval()
        logger.info(f'Now tracker is on device {next(model.parameters()).device}')

        # load ckpt
        ckpt_path = cfgs['motion_model_path']
        ckpt = torch.load(ckpt_path, map_location=device)
        logger.info(f'Loaded ckpt {ckpt_path}')
        model.load_state_dict(ckpt['model'])

        MambaTracklet.motion_predictor = model

    def __init__(self, cfgs, xywh, score, category, img_h, img_w, device):
        self.cfgs = cfgs

        # set manner, for mamba track, manner is diff, and for track_ssm, manner is bbox_and_diff
        self.manner = cfgs['manner']

        # initial position
        self._xywh = np.asarray(xywh, dtype=np.float32)  # match with precision of pytorch module
        self.is_activated = False

        self.score = score
        self.category = category

        # motion ssm predictor
        self.predicted_last_bbox = None  # the predicted bbox in last step

        # historical observation storage
        self.memo_bank = [self._xywh]
        self.diff_memo_bank = [np.zeros((4, ), dtype=np.float32)]  # store the diff between frames
        # self.diff_memo_bank[i + 1] = self.memo_bank[i + 1] - self.memo_bank[i]

        # store img size for norm
        self.img_size = [img_h, img_w]

        # set device
        self.device = device

    @property
    def tlwh(self, ):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.predicted_last_bbox is None:
            ret = self._xywh.copy()
        else:
            ret = self.predicted_last_bbox.copy()

        ret[:2] -= 0.5 * ret[2:]
        return ret

    def get_bbox(self, ):
        '''
        get tlwh for visualization
        '''
        if not len(self.memo_bank):
            ret = self._xywh.copy()            
        else:
            ret = self.memo_bank[-1].copy()
        
        ret[:2] -= 0.5 * ret[2:]
        return ret

    @torch.no_grad()
    def predict(self):
        pred = None
        if len(self.memo_bank) < self.cfgs['enable_time_thresh']:
            pred = self.memo_bank[-1]  # if only has one frame, return the initial position
        else:

            assert len(self.memo_bank) == len(self.diff_memo_bank)  

            # use the motion predictor
            hist_diff = np.array(self.diff_memo_bank[1:])

            # norm
            hist_diff[:, 0] /= self.img_size[1]
            hist_diff[:, 2] /= self.img_size[1]
            hist_diff[:, 1] /= self.img_size[0]
            hist_diff[:, 3] /= self.img_size[0]

            if self.manner == 'diff':

                hist_diff = torch.tensor(hist_diff * self.cfgs['scale_factor'], dtype=torch.float32).unsqueeze(0)  # (1, H, 4)                

                hist_diff = hist_diff.to(self.device)

                out = MambaTracklet.motion_predictor(hist_diff).squeeze()
                out = out.detach().cpu().numpy()

                # recover
                out[0] *= self.img_size[1]
                out[2] *= self.img_size[1]
                out[1] *= self.img_size[0]
                out[3] *= self.img_size[0]

                out /= self.cfgs['scale_factor']

                # add to last observation
                pred = deepcopy(self.memo_bank[-1])
                pred = pred + out

            elif self.manner == 'bbox_and_diff':

                hist = np.array(self.memo_bank[1:])

                # norm
                hist[:, 0] /= self.img_size[1]
                hist[:, 2] /= self.img_size[1]
                hist[:, 1] /= self.img_size[0]
                hist[:, 3] /= self.img_size[0]

                hist_bbox_and_diff = np.concatenate([hist * self.cfgs['scale_factor'], hist_diff * self.cfgs['scale_factor_diff']], axis=1)

                hist_bbox_and_diff = torch.tensor(hist_bbox_and_diff, dtype=torch.float32).unsqueeze(0)  # (1, H, 8)

                hist_bbox_and_diff = hist_bbox_and_diff.to(self.device)

                out = MambaTracklet.motion_predictor(hist_bbox_and_diff).squeeze()

                out = out.detach().cpu().numpy()

                # recover
                out[0] *= self.img_size[1]
                out[2] *= self.img_size[1]
                out[1] *= self.img_size[0]
                out[3] *= self.img_size[0]

                out /= self.cfgs['scale_factor']

                pred = out


        self.time_since_update += 1

        self.predicted_last_bbox = pred

    def activate(self, frame_id):
        self.track_id = self.next_id()

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.diff_memo_bank.append(new_track._xywh - self.memo_bank[-1])
        self.memo_bank.append(new_track._xywh)        

        if len(self.memo_bank) > self.cfgs['max_window']:
            self.memo_bank = self.memo_bank[1:]
            self.diff_memo_bank = self.diff_memo_bank[1:]

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):      

        if new_track is None:
            self.diff_memo_bank.append(self.predicted_last_bbox - self.memo_bank[-1])
            self.memo_bank.append(self.predicted_last_bbox)  # add last predicted bbox             
        else:
            self.frame_id = frame_id
            self.diff_memo_bank.append(new_track._xywh - self.memo_bank[-1])
            self.memo_bank.append(new_track._xywh)            
            self.score = new_track.score
        
        if len(self.memo_bank) > self.cfgs['max_window']:
            self.memo_bank = self.memo_bank[1:]
            self.diff_memo_bank = self.diff_memo_bank[1:]

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0
