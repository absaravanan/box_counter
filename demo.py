import torch
import detect
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
#weights
weights = "./visionModules/objectDetection/yolov5/runs/train/exp5/weights/best.pt"

# Images
image_filename = './box/obj_train_data/images/box_counting.png'  # or file, PIL, OpenCV, numpy, multiple

# Inference
prediction = detect.run(weights=weights,
                        source=image_filename)


img = mpimg.imread(prediction)
imgplot = plt.imshow(img)
plt.show()