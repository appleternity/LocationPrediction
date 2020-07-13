import os
import platform

if platform.node() == "lrs-khuang01.ist.psu.edu":
    root_dir = "/data/appleternity/LocationPrediction"
elif platform.node() == "appleternity-pc":
    root_dir = "/home/appleternity/workspace/lab/graph/baseline/release"
elif platform.node() == "dgx1":
    root_dir = "/dgxhome/czh5679/workspace/LocationPrediction"
else:
    print("unknown platform")
    quit()
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
