import easyocr
from PIL import Image
reader = easyocr.Reader(['ch_sim'])
text_list = reader.readtext('/raid/zxy/project/jddc/imgcls/angleClassifiationCode/data/晒单评论截图/ecae0a320bf5b12e13415925d342801c.jpg')
print(text_list)
img_text = ''
for text in text_list:
    if text[-1]>0.1:
        img_text = img_text + '。' + text[-2]
print(img_text)
