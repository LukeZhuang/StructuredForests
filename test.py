import os
import sys
from StructuredForests import StructuredForests

def bsds500_test(model, input_root, output_root):
    from skimage import img_as_float, img_as_ubyte
    from skimage.io import imread, imsave

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    image_dir = os.path.join(input_root, "BSDS500", "data", "images", "test")
    print(image_dir)
    file_names = list(filter(lambda name: name[-3:] == "jpg", os.listdir(image_dir)))
    print(file_names)

    n_image = len(file_names)

    for i, file_name in enumerate(file_names):
        print(file_name)
        img = img_as_float(imread(os.path.join(image_dir, file_name)))
        edge = img_as_ubyte(model.predict(img))

        imsave(os.path.join(output_root, file_name[:-3] + "png"), edge)

        sys.stdout.write("Processing Image %d/%d\r" % (i + 1, n_image))
        sys.stdout.flush()
    print()

options = {
        "rgbd": 0,
        "shrink": 2,
        "n_orient": 4,
        "grd_smooth_rad": 0,
        "grd_norm_rad": 4,
        "reg_smooth_rad": 2,
        "ss_smooth_rad": 8,
        "p_size": 32,
        "g_size": 16,
        "n_cell": 5,

        "n_pos": 10000,
        "n_neg": 10000,
        "fraction": 0.25,
        "n_tree": 8,
        "n_class": 2,
        "min_count": 1,
        "min_child": 8,
        "max_depth": 64,
        "split": "gini",
        "discretize": lambda lbls, n_class:
            discretize(lbls, n_class, n_sample=256, rand=rand),

        "stride": 2,
        "sharpen": 2,
        "n_tree_eval": 4,
        "nms": True,
    }
model = StructuredForests(options)

model.load_model()
bsds500_test(model, "toy", "edges")
