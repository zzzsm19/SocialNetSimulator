import abc
from environment.recommender.model import *
import importlib
import logging
import torch
import pandas as pd
from llm.prompt import get_prompt
from environment.recommender.model.DSSM import DSSM
import numpy as np



logger = logging.getLogger("MyLogger")

class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config, llm):
        self.config = config
        # self.page_size = config["page_size"]
        # module = importlib.import_module("recommender.model")
        # self.model = getattr(module, config["rec_model"])(config)
        self.model = DSSM(config)
        self.record = {}
        self.positive = {}
        self.inter_df = None
        self.inter_num=0
        self.llm = llm
        # for user in self.data.get_full_users():
        #     self.record[user] = []
        #     self.positive[user] = []

    # def get_full_sort_items(self, user):
    #     """
    #     Get a list of sorted items for a given user.
    #     """
    #     items = self.data.get_full_items()
    #     user_tensor = torch.tensor(user)
    #     items_tensor = torch.tensor(items)
    #     sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
    #     sorted_items = [item for item in sorted_items if item not in self.record[user]]
    #     sorted_item_names = self.data.get_item_names(sorted_items)
    #     description = self.data.get_item_description_by_id(sorted_items)
    #     items = [sorted_item_names[i] + ";;" + description[i] for i in range(len(sorted_item_names))]
    #     return items

    # def get_full_sort_items(self, user, cur_time, feed_num):
    #     items = self.data.get_current_simulated_content(cur_time)
    #     items_embeddings_dic = {}
    #     item_prompt_path = "llm/prompt_template/get_item_introduction.txt"
    #     for i, item in enumerate(items):
    #         item_prompt_input = item.copy()
    #         prompt = get_prompt(item_prompt_path, item_prompt_input)
    #         logger.debug(prompt)
    #         # result = self.llm.invoke(prompt)
    #         result = self.llm.embedding_invoke(prompt)
    #         logger.debug(result)
    #         items_embeddings_dic[i] = result
    #     user_introduction = user.get_introduction(0)
    #     user_embeddings = self.llm.embedding_invoke(user_introduction)
    #     sorted_items_dic = self.model.get_full_sort_items(user_embeddings, items_embeddings_dic)
    #     sorted_items_dic = list(sorted_items_dic.items())[:feed_num]
    #     sorted_items = []
    #     for i in range(feed_num):
    #         sorted_items.append(items[sorted_items_dic[i][0]])
    #     return sorted_items

    def get_full_sort_items(self, user, database, feed_num):
        items = []
        for data in database:
            items.extend(data)
        # print('dada')
        # items = self.data.get_current_simulated_content()
        print(items)
        if len(items) == 0:
            return []
        items_embeddings_dic = {}
        item_prompt_path = "llm/prompt_template/get_item_introduction.txt"
        for i, item in enumerate(items):
            item_prompt_input = item.copy()
            prompt = get_prompt(item_prompt_path, item_prompt_input)
            logger.debug(prompt)
            # result = self.llm.invoke(prompt)
            result = np.array(self.llm.embedding_invoke(prompt)['data']['embedding'])
            print('-'*100)
            print(prompt)
            print(result)
            logger.debug(result)
            items_embeddings_dic[i] = result
        user_introduction = user.get_introduction(-1)
        user_embeddings = np.array(self.llm.embedding_invoke(user_introduction)['data']['embedding'])
        sorted_items_dic = self.model.get_full_sort_items(user_embeddings, items_embeddings_dic)
        print('dagadaga')
        print(sorted_items_dic)
        sorted_items_dic = list(sorted_items_dic.items())[:feed_num]
        sorted_items = []
        for i in range(len(sorted_items_dic)):
            sorted_items.append(items[sorted_items_dic[i][0]])
        return sorted_items


    # def get_search_items(self, item_name):
    #     return self.data.search_items(item_name)
    #
    # def get_inter_num(self):
    #     return self.inter_num
    #
    # def update_history(self, user_id, item_names):
    #     """
    #     Update the history of a given user.
    #     """
    #     item_names = [item_name.strip(" <>'\"") for item_name in item_names]
    #     item_ids = self.data.get_item_ids(item_names)
    #     self.record[user_id].extend(item_ids)
    #
    # def update_positive(self, user_id, item_names):
    #     """
    #     Update the positive history of a given user.
    #     """
    #     item_ids = self.data.get_item_ids(item_names)
    #     if len(item_ids) == 0:
    #         return
    #     self.positive[user_id].extend(item_ids)
    #     self.inter_num+=len(item_ids)
    #
    # def save_interaction(self):
    #     """
    #     Save the interaction history to a csv file.
    #     """
    #     inters = []
    #     users = self.data.get_full_users()
    #     for user in users:
    #         for item in self.positive[user]:
    #             new_row = {"user_id": user, "item_id": item, "rating": 1}
    #             inters.append(new_row)
    #
    #         for item in self.record[user]:
    #             if item in self.positive[user]:
    #                 continue
    #             new_row = {"user_id": user, "item_id": item, "rating": 0}
    #             inters.append(new_row)
    #
    #     df = pd.DataFrame(inters)
    #     df.to_csv(
    #         self.config["interaction_path"],
    #         index=False,
    #     )
    #
    #     self.inter_df = df
