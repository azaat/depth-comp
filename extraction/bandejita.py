import csv
from src.rosbag_extraction_utils import make_dir_if_needed
import os
import shutil
import subprocess
from get_match import match
from src.alignment_utils import align_by_delta, _align


# Copyright 2022 Mobile Robotics Lab. at Skoltech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from src.alignment_utils import align_by_ref, align_by_delta, align_csv, align_imu


import tempfile

import cv2
# import ffmpeg
import numpy as np


def correct_rotation(frame, rotate_code):
    return cv2.rotate(frame, rotate_code)


class FrameMatcher:
    """
    Provides methods for matching frames in video with saved images
    """

    def __init__(self, video_filename, video_extension, capture_info_path, matching_threshold=0.99):
        self.video_filename = video_filename
        self.video_path = video_filename + "." + video_extension
        self.video = cv2.VideoCapture(self.video_path)
        self.capture_info_path = capture_info_path
        self.matching_threshold = matching_threshold

        timestamps_file = f"{self.capture_info_path}/{self.video_filename}_timestamps.csv"
        with open(timestamps_file, 'r') as timestamps_f:
            self.timestamps = timestamps_f.readlines()

    def split_to_frames(self, frame_dir):
        """
        Splits current video to frames and saves them in specified directory
        """

        # We need to check rotation metadata tag as android videos use it
        # to change video orientation
        rotate_code = self.check_rotation()
        success, image = self.video.read()
        count = 0
        print("Splitting video...")
        while success:
            if rotate_code is not None:
                image = correct_rotation(image, rotate_code)
            cv2.imwrite(f"{frame_dir}/frame{count}.jpg", image)  # save frame as JPEG file
            success, image = self.video.read()
            count += 1
        print("Finished splitting video")
        return count

    def match_frames(self):
        print(f"Matching frame images with threshold {self.matching_threshold}")
        with tempfile.TemporaryDirectory() as frame_dir:
            count = self.split_to_frames(frame_dir)
            assert len(
                self.timestamps) == count, f"Frames and timestamps counts do not match, {count}, {len(self.timestamps)}"

            flags = []
            step = 60
            for index, timestamp_str in enumerate(self.timestamps[::step]):
                i = index * step
                timestamp = timestamp_str.strip('\n')
                im1 = cv2.imread(f"{frame_dir}/frame{i}.jpg", cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(f"{self.capture_info_path}/{timestamp}.jpg", cv2.IMREAD_GRAYSCALE)
                # print(f"Matching {self.capture_info_path}/{timestamp}.jpg")

                # Resize video frame
                height1, width1 = im1.shape
                ratio1 = height1 / width1
                height2, width2 = im2.shape
                ratio2 = height2 / width2
                assert ratio1 == ratio2, "Image and frame ratio do not match"

                im1 = cv2.resize(im1, (width2, height2))

                # Match two images using cross-correlation
                res = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)
                flag = False
                if np.amax(res) > self.matching_threshold:
                    flag = True
                flags.append(flag)

            return all(flags)




    def check_rotation(self):
        """
        Checks video metadata for rotation tag
        :return: Optional rotation code
        """
        meta_dict = ffmpeg.probe(self.video_path)

        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotate_code = None
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotate_code = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        return rotate_code


def move_content(source_dir, target_dir):        
    file_names = os.listdir(source_dir)
    make_dir_if_needed(target_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)


def main():
    print("Extraction started")

    with open(
        './utils/matching.csv'
    ) as matching_file:
        i = 0
        matching_reader = csv.reader(matching_file)
        for row in matching_reader:
            print(f'{row[0]} - {row[2]}')
            b_path = '../bandejita/smartphones'

            if os.path.isdir(f'{b_path}/left/{row[0]}'):
                print('ARCORE')
            else:
                print('NORMAL')
        
            if i == 5:
                sm_dir = f'output/{row[2]}/smartphone_video_frames'
                # # make_dir_if_needed(sm_dir)

                # # # extract videos (for arcore - frames)
                threshold = 100000

                if os.path.isdir(f'{b_path}/left/{row[0]}'):
                    # move_content(f'{b_path}/left/{row[0]}', f'{sm_dir}/left')
                    print('Starting arcore extraction')
                    threshold = 15000000
                    time_ref = f'output/{row[2]}/_mcu_s10_ts/time_ref.csv'
                    with open(time_ref, 'r') as time_ref_file:
                        values = time_ref_file.readline().split(',')
                        seq = int(values[0])
                        ref_timestamp = int(values[1])
                        timestamp = int(values[2])
                        # obtain delta with the info from time reference file
                        delta = ref_timestamp - timestamp


                    target_dir = f'{sm_dir}/left'
                    filename_timestamps = list(map(
                        lambda x: (os.path.splitext(x)[0], int(os.path.splitext(x)[0])),
                        filter(
                            lambda x: os.path.splitext(x)[1] in ['.txt'],
                            os.listdir(target_dir)
                        )
                    ))
                    filename_timestamps.sort(key=lambda tup: tup[1])
                    _, extension = os.path.splitext(os.listdir(target_dir)[0])
                    _align(target_dir, filename_timestamps, ".txt", -delta)

                else:
                    print('Starting normal extraction')
                    session = subprocess.check_call(['./extract_sm.sh', f'{b_path}/left/VID/VID_{row[0]}.mp4', f'{sm_dir}/left'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # right extraction
                session = subprocess.check_call(['./extract_sm.sh', f'{b_path}/right/VID/VID_{row[1]}.mp4', f'{sm_dir}/right'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # align with azure
                align_by_delta(f'output/{row[2]}/_mcu_s10_ts/time_ref.csv', f'{sm_dir}/right')
                align_by_delta(f'output/{row[2]}/_mcu_s10_ts/time_ref.csv', f'{sm_dir}/left')
                
                # # match them (common step)
                match(f'{sm_dir}/left', f'{sm_dir}/right', f'{sm_dir}', threshold)

            i += 1

            
    

if __name__ == '__main__':
    main()
