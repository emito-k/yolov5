# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="1au9h1n9_0ut_10u6")
project = rf.workspace("capstone-fx07g").project("cheese-3st4y")
version = project.version(1)
dataset = version.download("yolov5")
