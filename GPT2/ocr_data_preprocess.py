# -*- coding: utf-8 -*-

import argparse
import re
import jieba
from tqdm import tqdm
import os
import json
import random

pattern_pun = '！，；：？、。"!,;:?."\''
pattern_jpg = re.compile(r'[A-Za-z0-9]+\.jpg')


def clean_text(text):

    #text = re.sub(r'[{}]+'.format(r'\d+\*\*'), '<num>', text)
    #text = re.sub(r'[{}]+'.format(r'\d+'), '<num>', text)
    text = re.sub('\/\/coupon.m.jd.com[\\S]*<url>', '<coupon>', text)

    # text = clean_punctuation(text)
    return text


def clean_punctuation(text):
    text = re.sub(r'[{}]+'.format(pattern_pun), '', text)
    return text.strip().lower()


def tokenize_spt(text):

    sp_token = ['<img>', '<url>', '<sos>', '<eos>', '<num>']

    resp_list = list()
    tmp_list = jieba.cut(text, cut_all=False)

    seg_list = list(tmp_list)
    i = 0

    while i < len(seg_list):
        if ''.join(seg_list[i:i + 3]) in sp_token:
            resp_list.append(''.join(seg_list[i:i + 3]))
            i = i + 3
        else:
            resp_list.append(''.join(seg_list[i]))
            i = i + 1

    return resp_list


class DataIterm(object):
    def __init__(self, sid, ques, ans, ctx):
        self.sid = sid
        self.ques = ques
        self.ans = ans
        self.ctx = ctx


def do_preprocess(dir, sess_turn):
    """
    :param dir:  官方数据存放路径
    :param sess_turn: context中保存的历史上下文的对话轮数
    :return: train_items, dev_items
             用于训练的train和dev数据，其中每条数据记录由以下几部分原始信息组成
             sid, 对话原始的session信息，后续按照需要可以根据该信息查询对话相关的知识库，本实例中未使用
             question, 该条训练数据所对应的用户的问题
             answer, 该条训练数据对应的客服的回答
             context, 该对话发生的上下文信息，该信息最大信息长度不超过sess_turn所定义的轮数
    """
    sess_len = sess_turn * 2
    train_items = list()
    dev_items = list()
    flag = False

    #for file, item_list in [('data_train.txt', train_items), ('data_dev.txt', dev_items)]:
    for file, item_list in [('data_train.txt', train_items), ('data_dev.txt', dev_items)]:  
        with open(dir+file) as f:
            lines = f.readlines()

        data_list = list()
        sess_pid = dict()
        for line in lines:
            word = line.strip().split('\t')
            sid = word[0]
            shop = word[1]
            pid = word[2]
            text = word[3]
            waiter = word[4]

            if pid:
                sess_pid[sid] = pid

            if waiter == '1':
                text = 'A:' + text
            else:
                text = 'Q:' + text

            data_list.append((sid, text))

        data_len = len(data_list)
        i = 0

        tmp_data_list = list()

        # 将原始数据按照session和问题、回答类型，
        # 用'|||'连接不同回车发送的内容
        while i < data_len:

            i_head = data_list[i][1][0]
            i_text = data_list[i][1]
            i_sid = data_list[i][0]

            j = i+1
            if j >= data_len:
                tmp_data_list.append((i_sid, i_text))
                break

            j_head = data_list[j][1][0]
            j_text = data_list[j][1]
            j_sid = data_list[j][0]

            add = 0
            while i_head == j_head and i_sid == j_sid:
                i_text = i_text + '|||' + j_text[2:]
                add = add + 1
                j = j + 1

                if j >= data_len:
                    break

                j_head = data_list[j][1][0]
                j_text = data_list[j][1]
                j_sid = data_list[j][0]

            i = i + add + 1
            tmp_data_list.append((i_sid, i_text))
        print("read done")
        # 遍历全部（session, Q:xxx） (session, A:xxx),
        # 构建训练输入文件，Q，A，Context，
        # 其中'@@@'间隔Context里面不同的Q或者A
        for idx, item in enumerate(tmp_data_list):

            sid = item[0]
            text = item[1]

            if text.startswith('A'):
                continue

            question = text.replace('Q:', '').strip()

            if question == '':
                continue

            if idx+1 >= len(tmp_data_list):
                continue

            n_item = tmp_data_list[idx+1]
            n_sid = n_item[0]
            if sid != n_sid:
                flag = True

            if idx+2<len(tmp_data_list):
                nn_item = tmp_data_list[idx+2]
                nn_sid = nn_item[0]
                if sid == nn_sid:
                    continue

            if not flag:
                n_text = n_item[1]

                answer = n_text.replace('A:', '').strip()

                if answer == '':
                    continue

                if idx > sess_len:
                    cand_data_list = tmp_data_list[idx-sess_len:idx]
                else:
                    cand_data_list = tmp_data_list[:idx]
            else:
                answer = tmp_data_list[idx-1][1].replace('A:', '').strip()
                question = tmp_data_list[idx-2][1].replace('Q:', '').strip()
                if answer in '' or question in '':
                    continue
                if idx-3 > sess_len:
                    cand_data_list = tmp_data_list[idx-3-sess_len:idx]
                else:
                    cand_data_list = tmp_data_list[:idx-3]

            contxt_list = list()
            for cand_item in cand_data_list:
                cand_sid = cand_item[0]
                cand_text = cand_item[1]

                if cand_sid != sid:
                    continue
                contxt_list.append(cand_text)

            context = '@@@'.join(contxt_list)

            item_list.append(DataIterm(sid, question, answer, context))
            flag = False
        print("concatation done")
    return train_items, dev_items

def get_ocr_res(ocr_phrase, ocr_text, keywords):
    img_text = ''
    flag = False
    for key in ocr_phrase.keys():
        if key in 'detail':
            for phrase in ocr_phrase[key]:
                if phrase in ocr_text:
                    img_text = img_text + ocr_text
                    flag = True
                    break
        elif key in 'combination':
            for phrase in ocr_phrase[key]:
                pl = phrase.split(' ')
                pattern = pl[0]+'[\\S]*'+pl[1]
                t = re.findall(pattern, ocr_text)
                if not t:
                    continue
                if '请输入' in t[0]:
                    img_text = img_text + ocr_text.replace(t[0], '')
                    flag = True
                    break
                else: img_text = img_text + t[0]
        elif key in 'extraction':
            for phrase in ocr_phrase[key]:
                if phrase in ocr_text:
                    img_text = img_text + phrase
        else :
            for phrase in ocr_phrase[key]:
                title = list(phrase.keys())[0]
                pl = title.split(' ')
                cnt = 0
                for p in pl:
                    if p in ocr_text:
                        cnt = cnt + 1
                if cnt > 2:
                    img_text = img_text + phrase[title]
        if flag:
            break
    if img_text in '':
        '''
        if len(ocr_text) <= 50:
            img_text = ocr_text
        else: 
            start = random.randint(0, len(ocr_text)-50)
            img_text = ocr_text[start:start+50]
        '''
        keyword_in_text = []    
        for k in keywords:
            index = ocr_text.find(k)
            if index != -1:
                keyword_in_text.append([k, index])
        if keyword_in_text:
            keyword_in_text.sort(key=lambda keyword_in_text: keyword_in_text[1])
            text = [x[0] for x in keyword_in_text]
            img_text = ''.join(text)
        else:
            img_text = '<img>'
    if img_text not in '<img>':
        img_text = '<S>' + img_text + '<E>'
    return img_text

def add_blank(text):
    text = text.replace(' ', '')
    text = re.sub("づ[^\u4e00-\u9fa5]*づ", "", text)
    text = re.sub('[★]*', '', text)
    text = text.replace('(*╹▽╹*)', '')
    text = text.replace('(✪▽✪)', '')
    text = text.replace('(/≧▽≦/)', '')
    text = text.replace('(*?▽?*)', '')
    text = text.replace('<img>', ' <img> ')
    text = text.replace('<url>', ' <url> ')
    text = text.replace('#E-s', ' #E-s ')
    text = text.replace('#crumb-wrap', ' #crumb-wrap ')
    text = text.replace('<coupon>', ' <coupon> ')
    text = text.replace('<S>', ' <S> ')
    text = text.replace('<E>', ' <E> ')
    return text

# topic_count = [0 for i in range(9)]
# def check_in(ls, x):
#     count = 0
#     for item in ls:
#         if item in x:
#             count += 1
#         if count >= 2:
#             return True
#     return False

def gen_train_dev_set(dir, train_items, dev_items, img_dir, ocr_path, img_label_path, keywords_path):
    f_train_out = open('./data_ocr/train.txt', 'w')
    f_dev_out = open('./data_ocr/dev.txt', 'w')

    ocr_res = {}
    for i in range(20):
        with open(ocr_path%i) as f:
            tmp = json.load(f)
            ocr_res.update(tmp)
    with open(img_label_path) as f:
        img_label = json.load(f)
    with open(keywords_path, 'r', encoding='utf-8-sig') as f:
        keywords = json.load(f)
    
    ocr_phrase = {
                    'detail':['交易纠纷详情', '取消/退款进度', '评价详情', '商品评价', '评价中心', '问题详情', 
                                '进度详情', '服务单详情', '投诉', '保单详情', '晒图相册'],
                    'combination': ['请输入 问题', '联系 客服', '已 发货', '价 保', '输入 手机号码'],
                    'extraction':['填写订单', '尺码', '运单详情', '电子存根', '签收底单', '发货清单', '增票资质', '交易物流',
                                    '订单详情', '等待收货', '自提点', '正在出库', '暂时无货', '送达时间', '支付成功', '付款成功',
                                    '等待付款', '正在配送途中', '错过售后申请时效', '已签收', '取件码', '不支持取消', '超出可购买总量',
                                    '京东收银台', '已限制购买数量', '超过了限购件数', '确认订单', '订单跟踪', '售后申请', '换货',
                                    '申请售后', '申请退款', '重新提交', '退换', '退换/售后', '取件', '售后', '上传快递单号', 
                                    '验证码', '保修期限', '已退款', '无货', '去结算', '审核未通过', '评价成功', '更改发货单',
                                    '未查到运单信息', '支付尾款', '选择售后类型', '购物车', '快递单号查询', '订购单', '新增收件人信息'
                                    '请输入正确的联系方式', '无货或不支持配送'],
                    'transformation': [{'退款明细 再次购买 卖了换钱 追加评价 查看发票 需付款': '订单截图'},
                                        {'颜色 数量 保障服务 确认': '加购物车选择配置'},
                                        {'全部分类 综合推荐 价格区间 筛选': '商品截图'}]
                }
    Cnt = 0
    for type in ['train', 'dev']:
        if type == 'train':
            items = train_items
            f_out = f_train_out
        elif type == 'dev':
            items = dev_items
            f_out = f_dev_out

        for item in tqdm(items):

            img_list = list()
            src_str = ''
            trg_str = ''

            ques = item.ques.strip()
            ans = item.ans.strip()
            ctx = item.ctx
            sid = item.sid

            ctx_list = ctx.split('@@@')

            for sent_i in ctx_list:
                if sent_i == '':
                    continue

                sent_i_type = sent_i[0]
                sent_i = sent_i[2:].strip()
                sent_i_list = sent_i.split('|||')

                for sent_j in sent_i_list:

                    if sent_j.endswith('.jpg'):
                        img_list.append(sent_j)
                        if sent_j in img_label.keys() and img_label[sent_j] in '0':
                            sent_j = '<S>商品截图<E>'
                        elif sent_j in img_label.keys() and img_label[sent_j] in '3':
                            sent_j = '<S>快递信息<E>'
                        elif sent_j in img_label.keys() and img_label[sent_j] in '1' and sent_j in ocr_res.keys() and len(ocr_res[sent_j]) > 4:
                            sent_j = get_ocr_res(ocr_phrase, ocr_res[sent_j], keywords)
                        else:
                            sent_j = '<img>'
                        #sent_j = '<img>'
                    else:
                        img_list.append('NULL')
                        sent_j = clean_text(sent_j)

                    #sent_seg = ' '.join(tokenize_spt(sent_j.strip()))
                    sent_seg = sent_j

                    if sent_seg:
                        src_str = src_str + sent_seg
                    else:
                        img_list.pop(-1)
                src_str = src_str + '\n'

            ques_list = ques.split('|||')
            ques_text = ''

            for sent in ques_list:
                if sent.endswith('.jpg'):
                    img_list.append(sent)
                    if sent in img_label.keys() and img_label[sent] in '0':
                        sent = '<S>商品截图<E>'
                    elif sent in img_label.keys() and img_label[sent] in '3':
                        sent = '<S>快递信息<E>'
                    elif sent in img_label.keys() and img_label[sent] in '1' and sent in ocr_res.keys() and len(ocr_res[sent]) > 4:
                        sent = get_ocr_res(ocr_phrase, ocr_res[sent], keywords)
                    else:
                        sent = '<img>'
                    #sent = '<img>'
                else:
                    img_list.append('NULL')
                    sent = clean_text(sent)

                sent = sent.strip()
                if sent:
                    #sent_seg = ' '.join(tokenize_spt(sent.strip()))
                    sent_seg = sent
                    src_str = src_str + sent_seg
                else:
                    img_list.pop(-1)
            ques_text = sent_seg
            src_str = src_str + '\n'

            ans_list = ans.split('|||')

            for sent in ans_list:
                if sent.endswith('jpg'):
                    if sent in img_label.keys() and img_label[sent] in '0':
                        sent = '<S>商品截图<E>'
                    elif sent in img_label.keys() and img_label[sent] in '3':
                        sent = '<S>快递信息<E>'
                    elif sent in img_label.keys() and img_label[sent] in '1' and sent in ocr_res.keys() and len(ocr_res[sent]) > 4:
                        sent = get_ocr_res(ocr_phrase, ocr_res[sent], keywords)
                    else:
                        sent = '<img>'
                    #sent = '<img>'
                else:
                    sent = clean_text(sent)

                #trg_str = trg_str + ' ' + ' '.join(tokenize_spt(sent.strip()))
                trg_str = trg_str + sent

            src_str = add_blank(src_str.strip())
            trg_str = add_blank(trg_str.strip())
            
            img_str = ' '.join(img_list)

            #src_list = src_str.split('</s>')

            #assert len(src_list) == len(img_list)

            # ls = [['订单', '配送', '请', '商品', '时间', '站点', '联系', '电话', '亲爱', '地址'],
            #       ['发票', '地址', '订单', '修改', '电子', '开具', '需要', '电话', '号', '姓名'],
            #       ['工作日', '退款', '订单', '帐', '取消', '申请', '支付', '成功', '商品', '请'],
            #       ['申请', '售后', '点击', '端', '提交', '客户服务', '审核', '链接', '返修', '补发'],
            #       ['订单', '起点', '时间', '日期', '下单', 'order', '编号', '催促', '信息', '订单号'],
            #       ['商品', '金额', '保价', '姓名', '手机', '申请', '快照', '订单', '查询', '请'],
            #       ['查询', '帮', '调货', '问题', '处理', '缺货', '订单号', '提供', '采购', '请'],
            #       ['！', '帮到', '谢谢', '支持', '感谢您', '评价', '客气', '妹子', '请', '祝您']]
            # x = ques_text + trg_str
            # for i, l in enumerate(ls):
            #     if check_in(l, x):
            #         topic_count[i] += 1
            # topic_count[-1] += 1

            f_out.write(sid+ '\n' +src_str+'\n'+trg_str+'\n\n')
            Cnt = Cnt + 1
    print(Cnt)
# import numpy
# a = [50440, 12557, 42205, 17641, 9861, 43475, 68308, 57471] #, 213644]
# a = numpy.array(a)
# print(numpy.sum(a))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process raw data")

    parser.add_argument('-d', '--directory', default='../data/')
    parser.add_argument('-i', '--img_dir', default='../data/image/')
    parser.add_argument('-o', '--ocr_path', default='./ocr/output_%d.json')
    parser.add_argument('-l', '--img_label_path', default='./label.json')
    parser.add_argument('-s', '--sess_turns', default=14)
    parser.add_argument('-k', '--keywords_path', default='./ocr/keywords.json')
    args = parser.parse_args()

    train_items, dev_items = do_preprocess(args.directory, args.sess_turns)
    print(len(train_items), len(dev_items))

    gen_train_dev_set(args.directory, train_items, dev_items, args.img_dir, args.ocr_path, args.img_label_path, args.keywords_path)

    # print(topic_count)