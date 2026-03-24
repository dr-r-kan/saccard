import os
import sys
import cv2
import math
from contextlib import contextmanager

# Must be set before importing mediapipe/tflite backends.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')

import mediapipe as mp
import numpy as np
from absl import logging as absl_logging
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.utils.cuda_utils import *
from tqdm import tqdm

absl_logging.set_verbosity(absl_logging.ERROR)


@contextmanager
def _suppress_native_stderr():
    """Compatibility wrapper; stderr is no longer suppressed."""
    yield

"""
This module defines classes or methods used for Signal extraction and processing.
"""

def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Convert normalized landmark coordinates to pixel coordinates."""
    def is_valid(value):
        return (value >= 0) and (value <= 1)
    if not (is_valid(normalized_x) and is_valid(normalized_y)):
        return None
    x_px = min(int(math.floor(normalized_x * image_width)), image_width - 1)
    y_px = min(int(math.floor(normalized_y * image_height)), image_height - 1)
    return x_px, y_px

def _landmark_is_valid(landmark, visibility_threshold=0.5, presence_threshold=0.5):
    """Check if a mediapipe landmark meets visibility and presence thresholds."""
    # HasField raises ValueError for non-optional proto3 scalar fields;
    # in proto3_optional fields (as used by MediaPipe), it works as expected.
    if landmark.HasField('visibility') and landmark.visibility < visibility_threshold:
        return False
    if landmark.HasField('presence') and landmark.presence < presence_threshold:
        return False
    return True

class SignalProcessing():
    """
        This class performs offline signal extraction with different methods:

        - holistic.

        - squared / rectangular patches.
    """

    def __init__(self):
        # Common parameters #
        self.tot_frames = None
        self.visualize_skin_collection = []
        self.skin_extractor = SkinExtractionConvexHull('CPU')
        # Patches parameters #
        high_prio_ldmk_id, mid_prio_ldmk_id = get_magic_landmarks()
        self.ldmks = high_prio_ldmk_id + mid_prio_ldmk_id
        self.square = None
        self.rects = None
        self.visualize_skin = False
        self.visualize_landmarks = False
        self.visualize_landmarks_number = False
        self.visualize_patch = False
        self.font_size = 0.3
        self.font_color = (255, 0, 0, 255)
        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []
        self.holistic_downsample_scale = 1.0
        self.holistic_landmark_refresh_stride = 1

    def set_holistic_speedup(self, downsample_scale=1.0, landmark_refresh_stride=1):
        """
        Configure CPU acceleration knobs for holistic extraction.

        Args:
            downsample_scale (float): Scale used for FaceMesh inference only.
                Values in (0, 1] reduce inference cost; 1.0 disables downsampling.
            landmark_refresh_stride (int): Run FaceMesh every N frames and reuse
                the most recent landmarks in-between.
        """
        scale = float(downsample_scale)
        if not np.isfinite(scale):
            scale = 1.0
        self.holistic_downsample_scale = float(min(max(scale, 0.25), 1.0))

        stride = int(landmark_refresh_stride)
        self.holistic_landmark_refresh_stride = max(1, stride)

    def choose_cuda_device(self, n):
        """
        Choose a CUDA device.

        Args:  
            n (int): number of a CUDA device.

        """
        select_cuda_device(n)

    def display_cuda_device(self):
        """
        Display your CUDA devices.
        """
        cuda_info()

    def set_total_frames(self, n):
        """
        Set the total frames to be processed; if you want to process all the possible frames use n = 0.
        
        Args:  
            n (int): number of frames to be processed.
            
        """
        if n < 0:
            print("[ERROR] n must be a positive number!")
        self.tot_frames = int(n)

    def set_skin_extractor(self, extractor):
        """
        Set the skin extractor that will be used for skin extraction.
        
        Args:  
            extractor: instance of a skin_extraction class (see :py:mod:`pyVHR.extraction.skin_extraction_methods`).
            
        """
        self.skin_extractor = extractor

    def set_visualize_skin_and_landmarks(self, visualize_skin=False, visualize_landmarks=False, visualize_landmarks_number=False, visualize_patch=False):
        """
        Set visualization parameters. You can retrieve visualization output with the 
        methods :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_skin` 
        and :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_patches`

        Args:  
            visualize_skin (bool): The skin and the patches will be visualized.
            visualize_landmarks (bool): The landmarks (centers of patches) will be visualized.
            visualize_landmarks_number (bool): The landmarks number will be visualized.
            visualize_patch (bool): The patches outline will be visualized.
        
        """
        self.visualize_skin = visualize_skin
        self.visualize_landmarks = visualize_landmarks
        self.visualize_landmarks_number = visualize_landmarks_number
        self.visualize_patch = visualize_patch

    def get_visualize_skin(self):
        """
        Get the skin images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly.
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_skin_collection

    def get_visualize_patches(self):
        """
        Get the 'skin+patches' images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly.
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_landmarks_collection

    def extract_raw(self, videoFileName):
        """
        Extracts raw frames from video.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            ndarray: raw frames with shape [num_frames, height, width, rgb_channels].
        """

        frames = []
        for frame in extract_frames_yield(videoFileName):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # convert to RGB

        return np.array(frames)

    ### HOLISTIC METHODS ###

    def extract_raw_holistic(self, videoFileName):
        """
        Locates the skin pixels in each frame. This method is intended for rPPG methods that use raw video signal.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: raw signal as float32 ndarray with shape [num_frames, rows, columns, rgb_channels].
        """

        skin_ex = self.skin_extractor

        mp_face_mesh = mp.solutions.face_mesh

        sig = []
        processed_frames_count = 0

        with _suppress_native_stderr():
            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                for frame in extract_frames_yield(videoFileName):
                    # convert the BGR image to RGB.
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frames_count += 1
                    width = image.shape[1]
                    height = image.shape[0]
                    # [landmarks, info], with info->x_center ,y_center, r, g, b
                    ldmks = np.zeros((468, 5), dtype=np.float32)
                    ldmks[:, 0] = -1.0
                    ldmks[:, 1] = -1.0
                    ### face landmarks ###
                    results = face_mesh.process(image)
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = [l for l in face_landmarks.landmark]
                        for idx in range(len(landmarks)):
                            landmark = landmarks[idx]
                            if _landmark_is_valid(landmark):
                                coords = _normalized_to_pixel_coordinates(
                                    landmark.x, landmark.y, width, height)
                                if coords:
                                    ldmks[idx, 0] = coords[1]
                                    ldmks[idx, 1] = coords[0]
                        ### skin extraction ###
                        cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                            image, ldmks)
                    else:
                        cropped_skin_im = np.zeros_like(image)
                        full_skin_im = np.zeros_like(image)
                    if self.visualize_skin == True:
                        self.visualize_skin_collection.append(full_skin_im)
                    ### sig computing ###
                    sig.append(full_skin_im)
                    ### loop break ###
                    if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                        break
        sig = np.array(sig, dtype=np.float32)
        return sig

    def extract_holistic(self, videoFileName):
        """
        This method compute the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_face_mesh = mp.solutions.face_mesh

        sig = []
        processed_frames_count = 0
        cached_ldmks = None
        cached_face_found = False
        scale = float(getattr(self, 'holistic_downsample_scale', 1.0))
        refresh_stride = int(getattr(self, 'holistic_landmark_refresh_stride', 1))

        total_frames = 0
        cap = cv2.VideoCapture(videoFileName)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if self.tot_frames is not None and self.tot_frames > 0:
            total_frames = min(total_frames, int(self.tot_frames)) if total_frames > 0 else int(self.tot_frames)
        pbar = None
        try:
            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc="RGB holistic frames",
                unit="fr",
                file=sys.stdout,
                disable=not bool(getattr(sys.stdout, "isatty", lambda: False)()),
            )
        except OSError:
            # Some spawned Windows subprocesses expose an invalid std handle.
            pbar = None

        with _suppress_native_stderr():
            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                for frame in extract_frames_yield(videoFileName):
                    # convert the BGR image to RGB.
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frames_count += 1
                    width = image.shape[1]
                    height = image.shape[0]
                    # [landmarks, info], with info->x_center ,y_center, r, g, b
                    ldmks = np.zeros((468, 5), dtype=np.float32)
                    ldmks[:, 0] = -1.0
                    ldmks[:, 1] = -1.0
                    do_refresh = (processed_frames_count == 1) or (cached_ldmks is None) or ((processed_frames_count - 1) % refresh_stride == 0)
                    if do_refresh:
                        ### face landmarks ###
                        if scale < 0.999:
                            infer_image = cv2.resize(
                                image,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_AREA,
                            )
                        else:
                            infer_image = image
                        results = face_mesh.process(infer_image)
                        if results.multi_face_landmarks:
                            face_landmarks = results.multi_face_landmarks[0]
                            landmarks = [l for l in face_landmarks.landmark]
                            for idx in range(len(landmarks)):
                                landmark = landmarks[idx]
                                if _landmark_is_valid(landmark):
                                    coords = _normalized_to_pixel_coordinates(
                                        landmark.x, landmark.y, width, height)
                                    if coords:
                                        ldmks[idx, 0] = coords[1]
                                        ldmks[idx, 1] = coords[0]
                            cached_ldmks = ldmks.copy()
                            cached_face_found = True
                        else:
                            cached_ldmks = None
                            cached_face_found = False

                    if cached_face_found and cached_ldmks is not None:
                        ldmks = cached_ldmks
                        ### skin extraction ###
                        cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                            image, ldmks)
                    else:
                        cropped_skin_im = np.zeros_like(image)
                        full_skin_im = np.zeros_like(image)
                    if self.visualize_skin == True:
                        self.visualize_skin_collection.append(full_skin_im)
                    ### sig computing ###
                    sig.append(holistic_mean(
                        cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                    if pbar is not None:
                        pbar.update(1)
                    ### loop break ###
                    if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                        break
        if pbar is not None:
            pbar.close()
        sig = np.array(sig, dtype=np.float32)
        return sig

    ### PATCHES METHODS ###

    def set_landmarks(self, landmarks_list):
        """
        Set the patches centers (landmarks) that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            landmarks_list (list): list of positive integers between 0 and 467 that identify patches centers (landmarks).
        """
        if not isinstance(landmarks_list, list):
            print("[ERROR] landmarks_set must be a list!")
            return
        self.ldmks = landmarks_list

    def set_square_patches_side(self, square_side):
        """
        Set the dimension of the square patches that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            square_side (float): positive float that defines the length of the square patches.
        """
        if not isinstance(square_side, float) or square_side <= 0.0:
            print("[ERROR] square_side must be a positive float!")
            return
        self.square = square_side

    def set_rect_patches_sides(self, rects_dim):
        """
        Set the dimension of each rectangular patch. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            rects_dim (float32 ndarray): positive float32 np.ndarray of shape [num_landmarks, 2]. If the list of used landmarks is [1,2,3] 
                and rects_dim is [[10,20],[12,13],[40,40]] then the landmark number 2 will have a rectangular patch of xy-dimension 12x13.
        """
        if type(rects_dim) != type(np.array([])):
            print("[ERROR] rects_dim must be an np.ndarray!")
            return
        if rects_dim.shape[0] != len(self.ldmks) and rects_dim.shape[1] != 2:
            print("[ERROR] incorrect rects_dim shape!")
            return
        self.rects = rects_dim

    def extract_patches(self, videoFileName, region_type, sig_extraction_method):
        """
        This method compute the RGB-mean signal using specific skin regions (patches).

        Args:
            videoFileName (str): video file name or path.
            region_type (str): patches types can be  "squares" or "rects".
            sig_extraction_method (str): RGB signal can be computed with "mean" or "median". We recommend to use mean.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, num_patches, rgb_channels].
        """
        if self.square is None and self.rects is None:
            print(
                "[ERROR] Use set_landmarks_squares or set_landmarkds_rects before calling this function!")
            return None
        if region_type != "squares" and region_type != "rects":
            print("[ERROR] Invalid landmarks region type!")
            return None
        if sig_extraction_method != "mean" and sig_extraction_method != "median":
            print("[ERROR] Invalid signal extraction method!")
            return None

        ldmks_regions = None
        if region_type == "squares":
            ldmks_regions = np.float32(self.square)
        elif region_type == "rects":
            ldmks_regions = np.float32(self.rects)

        sig_ext_met = None
        if sig_extraction_method == "mean":
            if region_type == "squares":
                sig_ext_met = landmarks_mean
            elif region_type == "rects":
                sig_ext_met = landmarks_mean_custom_rect
        elif sig_extraction_method == "median":
            if region_type == "squares":
                sig_ext_met = landmarks_median
            elif region_type == "rects":
                sig_ext_met = landmarks_median_custom_rect

        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

        skin_ex = self.skin_extractor

        mp_face_mesh = mp.solutions.face_mesh

        sig = []
        processed_frames_count = 0
        self.patch_landmarks = []
        self.cropped_skin_im_shapes = [[], []]
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                magic_ldmks = []
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if _landmark_is_valid(landmark):
                            coords = _normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                                
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, ldmks)

                    self.cropped_skin_im_shapes[0].append(cropped_skin_im.shape[0])
                    self.cropped_skin_im_shapes[1].append(cropped_skin_im.shape[1])

                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                    self.cropped_skin_im_shapes[0].append(cropped_skin_im.shape[0])
                    self.cropped_skin_im_shapes[1].append(cropped_skin_im.shape[1])

                ### sig computing ###
                for idx in self.ldmks:
                    magic_ldmks.append(ldmks[idx])
                magic_ldmks = np.array(magic_ldmks, dtype=np.float32)
                temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                   np.int32(SignalProcessingParams.RGB_LOW_TH), 
                                   np.int32(SignalProcessingParams.RGB_HIGH_TH))
                sig.append(temp)

                # save landmarks coordinates
                self.patch_landmarks.append(magic_ldmks[:,0:3])

                # visualize patches and skin
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                if self.visualize_landmarks == True:
                    annotated_image = full_skin_im.copy()
                    color = np.array([self.font_color[0],
                                      self.font_color[1], self.font_color[2]], dtype=np.uint8)
                    for idx in self.ldmks:
                        cv2.circle(
                            annotated_image, (int(ldmks[idx, 1]), int(ldmks[idx, 0])), radius=0, color=self.font_color, thickness=-1)
                        if self.visualize_landmarks_number == True:
                            cv2.putText(annotated_image, str(idx),
                                        (int(ldmks[idx, 1]), int(ldmks[idx, 0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_size,  self.font_color,  1)
                    if self.visualize_patch == True:
                        if region_type == "squares":
                            sides = np.array([self.square] * len(magic_ldmks))
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), sides, sides, color)
                        elif region_type == "rects":
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), np.array(self.rects[:, 0]), np.array(self.rects[:, 1]), color)
                    self.visualize_landmarks_collection.append(
                        annotated_image)
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return np.copy(sig[:, :, 2:])

    def get_landmarks(self):
        """
        Returns landmarks ndarray with shape [num_frames, num_estimators, 2-coords] or empty array 
        """
        if hasattr(self, 'patch_landmarks'):
            return np.array(self.patch_landmarks)
        else:
            return np.empty(0)

    def get_cropped_skin_im_shapes(self):
        """
        Returns cropped skin shapes with shape [height, width, rgb] or empty array
        """
        if hasattr(self, "cropped_skin_im_shapes"):
            return np.array(self.cropped_skin_im_shapes)
        else:
            return np.empty((0, 0, 0))


