import os
import Augmentor


if __name__ =="__main__":
    path = '{0}/dataset/endoscopy/JFD-03K/jfd_dataset/d1.9.2/'.format(
        os.getcwd()[:os.getcwd().find('/python/')
    aug_path = '{0}/dataset/endoscopy/JFD-03K/jfd_dataset/aug5_3/'.format(
        os.getcwd()[:os.getcwd().find('/python/')

    sites = os.listdir(path)
    for site in sites:
        p = Augmentor.Pipeline(source_directory = path+site, output_directory= aug_path+site)
        p.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)
        p.skew(probability=0.9, magnitude=0.6)
        p.random_distortion(probability=1.0, grid_height=5, grid_width=5, magnitude=5)
        p.random_brightness(probability=1.0, min_factor=0.5, max_factor=1.2)
        p.sample(2000)