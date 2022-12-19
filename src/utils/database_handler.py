import pymongo
import os
from from_root import from_root
from pymongo import MongoClient
from src.entity.config_entity import DatabaseConfig

class  MongoDBClient:

    def __init__(self) -> None:
        self.db_config = DatabaseConfig()
        print(self.db_config.URL)
        print(self.db_config.USERNAME)
        print(self.db_config.PASSWORD)
        url = self.db_config.URL.replace("<username>", self.db_config.USERNAME).replace("<password>", self.db_config.PASSWORD)
        
        self.client = MongoClient(url)

    def insert_bulk_record(self, records):
        try:
            # check if collection exists or not
            # if collection not exists then create it
            db = self.client[self.db_config.DBNAME]
            collection = self.db_config.COLLECTION
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            
            results = db[collection].insert_many(records)
            return {"Response": "Success", "Inserted Documents": len(results.inserted_ids)}
        except Exception as e:
            raise e

    def get_collection_documents(self):
        try:
            db = self.client[self.db_config.DBNAME]
            collection = self.db_config.COLLECTION
            result = db[collection].find()
            return {"response":"Success", "Info":result}


        except Exception as exp:
            raise exp
    

    def drop_collection(self):
        try:
            db = self.client[self.db_config.DBNAME]
            collection = self.db_config.COLLECTION
            db[collection].drop()
            return {"response":"Success"}
        except Exception as exp:
            raise exp


if __name__ == "__main__":
    data = [
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"}
    ]

    mongo = MongoDBClient()
    print(mongo.insert_bulk_record(data))
    print(mongo.drop_collection())
    result = mongo.get_collection_documents()
    print(result["Info"])
