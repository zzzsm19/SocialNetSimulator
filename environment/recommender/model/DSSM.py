import importlib
from typing import Union, List, Optional
# from model.base import BaseModel
import numpy as np
import torch
# import random
import collections

class BaseModel(object):
    """Base class for all models."""

    def __init__(self, config):
        self.config = config
        self.items = None

    def get_full_sort_items(self, user_id, *args, **kwargs):
        """Get a list of sorted items for a given user."""
        raise NotImplementedError

    def _sort_full_items(self, user_id, *args, **kwargs):
        """Sort a list of items for a given user."""
        raise NotImplementedError

class Random_model(BaseModel):
    def __init__(self, config):
        self.config = config

    def get_full_sort_items(self, users, items):
        """Get a list of sorted items for a given user."""

        sorted_items = self._sort_full_items(users, items)
        return sorted_items

    def _sort_full_items(self, user, items):
        """Return a random list of items for a given user."""
        random_items = torch.randperm(items.size(0)).tolist()
        return random_items

class DSSM(BaseModel):
    def __init__(self, config):
        self.config = config

    def get_full_sort_items(self, user_embeddings, items_embeddings_dic):

        items_embeddings = torch.from_numpy(np.stack(list(items_embeddings_dic.values()), 0))
        # user_embeddings = torch.from_numpy(user_embeddings).reshape(1, -1)
        # dot_product = torch.matmul(user_embeddings, items_embeddings.T)
        # normalized_dot_product = torch.nn.functional.normalize(dot_product, p = 2, dim = 1)
        # print(normalized_dot_product.shape)
        # print(items_embeddings.shape)
        user_embeddings = torch.from_numpy(np.repeat(user_embeddings.reshape(1, -1), items_embeddings.shape[0], 0))

        cosine_similarity = torch.nn.functional.cosine_similarity(user_embeddings, items_embeddings, dim = 1).reshape(-1,1).view(-1)
        print('aa',cosine_similarity)
        sorted_items_dic = {}
        for i, (key, _) in enumerate(items_embeddings_dic.items()):
            sorted_items_dic[key] = cosine_similarity[i].item()

        sorted_items_dic = collections.OrderedDict(sorted(sorted_items_dic.items(), key = lambda x: x[1], reverse = True))
        return sorted_items_dic



# a = {'1':np.array([0,0], dtype=float),'2':np.array([3,4], dtype=float)}
# a = collections.OrderedDict(a)
# print(list(a.items())[:1])
# b = np.array([1,1], dtype=float)
#
# c = DSSM(None)
# print(c.get_full_sort_items(b,a))

# a = np.array([1,1,1]).reshape(1,-1)
#
# print(np.repeat(a,2,0))

