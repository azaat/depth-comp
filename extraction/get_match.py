# Copyright 2021 Mobile Robotics Lab. at Skoltech
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

import glob
import os
import pandas as pd


def match(vid_1, vid_2, out_d, threshold):
    out_images_1 = sorted(sum([glob.glob(f'{vid_1}/*{e}') for e in ['.jpg', '.png']], []))
    out_images_2 = sorted(sum([glob.glob(f'{vid_2}/*{e}') for e in ['.jpg', '.png']], []))

    image_timestamps_1 = sorted(list(map(
        lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        out_images_1)))
    image_timestamps_2 = sorted(list(map(
        lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        out_images_2)))

    THRESHOLD_NS = threshold

    left = pd.DataFrame({'t': image_timestamps_1,
                         'left': image_timestamps_1}, dtype=int)
    # TODO: change this quick hack to prevent pandas from
    # converting ints to floats
    right = pd.DataFrame({'t': image_timestamps_2,
                          'right_int': image_timestamps_2,
                          'right': list(map(str, image_timestamps_2))},
                         )
    print(right.dtypes)
    # align by nearest, because we need to account for frame drops
    df = pd.merge_asof(left, right, on='t',
                       tolerance=THRESHOLD_NS,
                       allow_exact_matches=True,
                       direction='nearest')
    df = df.dropna()
    df = df.drop('t', axis='columns')
    df = df.drop('right_int', axis='columns')

    df = df.reset_index(drop=True)
    print(df.head())
    print(df.dtypes)

    df.to_csv(f'{out_d}/match.csv')