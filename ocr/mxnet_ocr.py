from cnstd import CnStd
from cnocr import CnOcr
import time
import os
import re
import json
from tqdm import tqdm
from multiprocessing import Process

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_CUDA_LIB_CHECKING'] = '0'

# 需要改成自己的目录
prefix = '../online_test_data/images_test/'


ls = os.listdir(
    prefix
)
# ls = ls[:1030]

time_start = time.time()

ls_len = len(ls)

# 改成你期望的个数，V100的显存32G，远大于1080ti的12G，我感觉可以改成4 * 4 * (32 / 12) = 40
# 提交的时候由于只有一张V100，因此可以改成 4 * (32 / 12) = 10
block_num = 10
block_len = ls_len // block_num
block_start = [i * block_len for i in range(block_num)]
block_end = [(i + 1) * block_len for i in range(block_num)]
block_end[-1] = ls_len


def target_function(idx, ls, prefix):
    # 使得每一张卡都能被充分用到
    # 提交的时候由于只有一张卡，需要全部都改成0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    std = CnStd(context='gpu')
    cn_ocr = CnOcr(context='gpu')
    result = dict()
    for file_name in tqdm(ls):
        file_path = os.path.join(prefix, file_name)
        box_info_list = std.detect(file_path)
        output = ''
        for box_info in box_info_list:
            cropped_img = box_info['cropped_img']  # 检测出的文本框
            if type(cropped_img) != type(None):
                ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
                output += ''.join(ocr_res)
            # print('ocr result: %s' % ''.join(ocr_res))
        output = output.replace(' ', '')
        output = re.sub("[^\u4e00-\u9fa5]", "", output)
        result[file_name] = output
    with open('./output_%d.json' % (idx), 'w', encoding='utf-8') as w:
        w.write(json.dumps(result, ensure_ascii=False, indent=2))

processes = [
    Process(target=target_function, args=(i, ls[block_start[i]:block_end[i]], prefix))
    for i in range(block_num)
]

for p in processes:
    p.start()
for p in processes:
    p.join()

time_end = time.time()
print('time cost', time_end - time_start, 's')