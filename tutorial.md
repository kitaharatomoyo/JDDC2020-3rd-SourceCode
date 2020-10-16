1 训练图片分类模型（商品截图、快递、需要进行ocr、其他）
cd imgcls
python train.py
保存路径：checkpoints/ocr/ResNet50.pt

2 ocr
cd ocr
cd img_emb
python get_label.py
得到图片标签（label.json）
将python3.6/dist-packages/cnocr/cn_ocr.py 替换为该目录下的cn_ocr.py
python mxnet_ocr.py
得到图片ocr结果（output_*.json

3 训练生成模型
cd GPT2
python ocr_data_preprocess.py
python train.py

