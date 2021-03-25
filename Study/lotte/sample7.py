import glob
import os
import numpy as np
from PIL import Image

# 코드 실행시 모든 파일에 000을 붙여준다!
# for i in range(1000):
#     os.mkdir('../study/LPD_competition/train_new/{0:04}'.format(i))

#     for img in range(48):
#         image = Image.open(f'../study/LPD_competition/train/{i}/{img}.jpg')
#         image.save('../study/LPD_competition/train_new/{0:04}/{1:02}.jpg'.format(i, img))

for i in range(72000):
    image = Image.open(f'../study/LPD_competition/test/{i}.jpg')
    image.save('../study/LPD_competition/test_new/test_new/{0:05}.jpg'.format(i))