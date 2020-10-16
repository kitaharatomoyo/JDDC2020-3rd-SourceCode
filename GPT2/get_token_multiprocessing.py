from multiprocessing import Process
from transformers import BertTokenizer
import argparse
from tqdm import tqdm

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data_ocr/dev.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data_ocr/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--train_mmi', action='store_true', help="若指定该参数，则训练DialoGPT的MMI模型")
    parser.add_argument('--n_ctx', type=int, default=1024, required=False, help="最长序列长度")
    parser.add_argument('--train_mmi_tokenized_path', default='data_ocr/dev_mmi_tokenized.txt', type=str,
                        required=False,help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
    return parser.parse_args()

def encoder(index, train_data, args, tokenizer):
    with open(args.train_tokenized_path, "a+", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in dialogue:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances[1:]:
                #dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.extend(tokenizer.encode(utterance))
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            #dialogue_ids = dialogue_ids[-args.n_ctx:]
            dialogue_ids[0] = tokenizer.cls_token_id
            #assert len(dialogue_ids) <= 1024, '长度：'+utterance
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            f.write('\n')

def preprocess_raw_data(args, tokenizer):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    train_data = train_data[:-1]
    n_processes = 20 #number of processes
    n_total = len(train_data)
    length = float(n_total) / float(n_processes)
    indices = [int(round(i* length)) for i in range(n_processes)] + [n_total]

    sublists = [train_data[indices[i]:indices[i+1]] for i in range(n_processes)]
    processes = [Process(target=encoder,args=(i, x, args, tokenizer)) for i,x in enumerate(sublists)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

def mmi_encoder(index, train_data, args, tokenizer):
    with open(args.train_mmi_tokenized_path, "a+", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in dialogue:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in reversed(utterances[1:]):  # 将一段对话进行翻转
                #dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.extend(tokenizer.encode(utterance))
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            #dialogue_ids = dialogue_ids[:args.n_ctx]
            dialogue_ids[0] = tokenizer.cls_token_id
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            f.write('\n')

def preprocess_mmi_raw_data(args, tokenizer):
    """
    对原始语料进行处理，将原始语料的每段对话进行翻转，然后转换为用于train MMI模型的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance N[SEP]utterance N-1[SEP]utterance N-2[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    n_processes = 20 #number of processes
    n_total = len(train_data)
    length = float(n_total) / float(n_processes)
    indices = [int(round(i* length)) for i in range(n_processes)] + [n_total]

    sublists = [train_data[indices[i]:indices[i+1]] for i in range(n_processes)]
    processes = [Process(target=mmi_encoder,args=(i, x, args, tokenizer)) for i,x in enumerate(sublists)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

def main():
    args = setup_train_args()
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    tokenizer.add_special_tokens({'additional_special_tokens':['#E-s', '#crumb-wrap', '<coupon>', '<url>', '<img>', '<S>', '<E>']})
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    if args.train_mmi:  # 如果当前是要训练MMI模型
        preprocess_mmi_raw_data(args, tokenizer)
    else:  # 如果当前是要训练对话生成模型
        preprocess_raw_data(args, tokenizer)


if __name__=="__main__":
    main()
