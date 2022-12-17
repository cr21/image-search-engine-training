
from src.utils.s3_handler import s3Connection
from src.pipeline.searchpipeline import Pipeline

pipe = Pipeline()
if __name__=='__main__':
    print("app pipelien",__name__)