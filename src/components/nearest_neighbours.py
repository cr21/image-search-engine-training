from annoy import AnnoyIndex
from src.utils.database_handler import MongoDBClient
from typing_extensions import Literal
from tqdm import tqdm
import json
from src.entity.config_entity import AnnoyConfig

# https://sds-aau.github.io/M3Port19/portfolio/ann/
# from annoy import AnnoyIndex
# import random

# f = 40
# t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
# for i in range(1000):
#     v = [random.gauss(0, 1) for z in range(f)]
#     t.add_item(i, v)

# t.build(10) # 10 trees
# t.save('test.ann')

# # ...

# u = AnnoyIndex(f, 'angular')
# u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

class customAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]) -> None:
        """"
        @params:
            f : int -> Embedding size for Vector we are building tree for
            metric -> distance metric to use in Annoy Tree
        """
        super().__init__(f, metric)
        self.labels=[]

    def add_item(self, i: int, vector, label: str) -> None:
        super().add_item(i, vector)
        self.labels.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        """
        Annoy Index returns only index of nearest neighbors 
        but we want to show the image, so we can avoid the API call by just modifying this
        function to return link of image not just indexes
        
        """
        indexes  = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels =  [self.labels[link] for link in indexes]

        return labels


    def load(self, fn: str, prefault: bool = ...):
        super().load(fn)
        path = fn.replace(".ann",".json")
        self.labels=json.load(open(path, "r"))


    def save(self, fn: str, prefault: bool = ...):
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.labels, open(path, "w"))






class Annoy:
    """
    Custom Wrapper around Spotify Annoy Index
    
    """
    def __init__(self) -> None:
        self.config = AnnoyConfig()
        self.mongo_client = MongoDBClient()
        print(self.mongo_client.get_collection_documents())
        self.result = self.mongo_client.get_collection_documents()["Info"]
        
    def build_AnnoyFormat(self):

    
        Ann = customAnnoy(self.config.EMBEDDING_SIZE, 'euclidean')
        print("Creating ANN for predictions")
        # for i, record in tqdm(enumerate(self.result),total=8851):
        for i, record in tqdm(enumerate(self.result),total=8851):
            Ann.add_item(i,record["images"],record["s3_link"])
        
        # n_trees
        Ann.build(100)
        Ann.save(self.config.EMBEDDING_STORE_PATH)
        return True


    def run_step(self):
        self.build_AnnoyFormat()


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()        





