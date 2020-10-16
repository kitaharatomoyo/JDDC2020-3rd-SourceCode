from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    """

    """

    def __init__(self, data_list, n_ctx, train_mmi):
        self.data_list = data_list
        self.n_ctx = n_ctx
        self.train_mmi = train_mmi

    def __getitem__(self, index):
        input_ids = self.data_list[index].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        if self.train_mmi:
            type = 1
            flag = False
            img_type = 2
            pos = 0
            turn = 0
            token_type_ids = []
            position_ids = []
            turn_ids = []
            for id in input_ids:
                if id == 13322:
                    flag = True
                    continue
                if id == 13323:
                    flag = False
                    continue
                if flag:
                    token_type_ids.append(img_type+type)
                else:
                    token_type_ids.append(type)
                position_ids.append(pos)
                if pos < 1023:
                    pos = pos + 1
                turn_ids.append(turn)
                if id == 102:
                    type = 1 - type
                    pos = 0
                    if turn < 59:
                        turn = turn + 1
            input_ids = [i for i in input_ids if i != 13323 and i != 13322]
            input_ids = input_ids[:self.n_ctx]
            token_type_ids = token_type_ids[:self.n_ctx]
            position_ids = position_ids[:self.n_ctx]
            turn_ids = turn_ids[:self.n_ctx]
        else:
            type = 0
            flag = False
            img_type = 2
            turn = 60
            token_type_ids = []
            position_ids = []
            turn_ids = []
            for id in reversed(input_ids):
                if id == 13323:
                    flag = True
                    continue
                if id == 13322:
                    flag = False
                    continue                   
                if id == 102:
                    type = 1 - type
                    if turn > 0:
                        turn = turn - 1
                if flag:
                    token_type_ids.append(img_type+type)
                else: 
                    token_type_ids.append(type)
                turn_ids.append(turn)
            pos = 0
            for id in input_ids:
                if id == 13322 or id == 13323:
                    continue
                position_ids.append(pos)
                if pos < 1023:
                    pos = pos + 1
                if id == 102:
                    pos = 0
            token_type_ids.reverse()
            turn_ids.reverse()
            input_ids = [i for i in input_ids if i != 13322 and i != 13323]
            input_ids = input_ids[-self.n_ctx:]
            position_ids = position_ids[-self.n_ctx:]
            token_type_ids = token_type_ids[-self.n_ctx:]
            turn_ids = turn_ids[-self.n_ctx:]
        return {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'position_ids':position_ids, 'turn_ids':turn_ids}

    def __len__(self):
        return len(self.data_list)
