import random
from pymongo import MongoClient


class DMNBasic:
    def __init__(self, timestep=5, batch_size=12, iteration=30):
        client = MongoClient('localhost', 27017)
        db = client.djia_news_dataset
        self.db_tbl_price_news = db.price_news
        self.timestep = timestep
        self.batch_size = batch_size
        self.iteration = iteration
        self.data_num_row = 1989
        self.train_start_idx = [(x * 5) for x in range(self.batch_size * self.iteration)]
        self.valid_start_idx = self.batch_size * self.iteration * self.timestep
        self.evalu_start_idx = self.valid_start_idx + (self.data_num_row - (self.valid_start_idx)) // 2 + 1
        self.train_start_idx_list = self.train_start_idx.copy()
        self.valid_start_idx_list = [x for x in range(self.valid_start_idx, self.evalu_start_idx-timestep)]
        self.evalu_start_idx_list = [x for x in range(self.evalu_start_idx, self.data_num_row-timestep)]

    def _get_train_dataseq(self):
        """ retrieve a sequential dataset starting at a random index """
        timestep, batch_size = self.timestep, self.batch_size
        dataseq_x, dataseq_y = [], []
        if len(self.train_start_idx_list) == 0:
            self.train_start_idx_list = self.train_start_idx.copy()
        random.shuffle(self.train_start_idx_list)
        for _ in range(batch_size):
            idx = self.train_start_idx_list.pop()
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            dataseq_x.append([r for r in res_x])
            dataseq_y.append(res_y['Adj Close'])
        return dataseq_x, dataseq_y

    def _get_dataseq(self, type='validation'):
        """ retrieve dataset for validation or evaluation """
        timestep = self.timestep
        dataseq_x, dataseq_y = [], []
        if type == 'validation':
            idx_list = self.valid_start_idx_list
        elif type == 'evaluation':
            idx_list = self.evalu_start_idx_list
        else:
            raise Exception('unsupported dataseq type')
        for idx in idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            dataseq_x.append([r for r in res_x])
            dataseq_y.append(res_y['Adj Close'])
        return dataseq_x, dataseq_y
