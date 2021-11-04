import os
import cv2
import math
import argparse

import tensorflow as tf

from tqdm import tqdm

from dataset import load_image
from evaluate import classify
from models import select_model

from tensorflow.keras.utils import Sequence

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', type=str, help='select gpu to train', default="0")
    parser.add_argument('--video_path', type=str, default="", help='path to video')
    parser.add_argument('--ckpt_path', type=str, default="", help='checkpoint path to load')
    parser.add_argument('--batch', type=int, default=256, help='set batch size')
    parser.add_argument('--scale',  action="store_true", help='scale in preprocess')

    return parser.parse_args()


class EGDVideoDataset(Sequence):
    def __init__(self, root_path, save_dir, batch_size, image_shape, scale, crop_size=(0,0,0,0)):
        # self.x = self.save_and_crop(root_path, crop_size)
        self.frame_path = root_path
        # self.x = self.save_and_crop_4fps(root_path, crop_size)
        self.x = self.save_frames(root_path, save_dir)
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.scale = scale

    def save_frames(self, video_path, save_dir):
        X = []
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 1
            pbar = tqdm(total = frame_count)

            count = 0
            while cap.isOpened():
                success, image = cap.read()
                if image is None:
                    print('Completed!')
                    break
                cv2.imwrite(save_dir + "frame%d.jpg" % count, image)
                pbar.update(i)
                count += 1
    
        for file in tqdm(os.listdir(save_dir)):
            X.append(os.path.join(save_dir, file))
        
        return X

    def save_and_crop(self, video_path, crop_size):
        save_dir = video_path[:-4] + '/'
        X = []
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 1
            pbar = tqdm(total = frame_count)

            count = 0
            while cap.isOpened():
                success, image = cap.read()
                if image is None:
                    print('Completed!')
                    break
                crop_image = image[crop_size[1]:-crop_size[3],crop_size[0]:-crop_size[2]]
                cv2.imwrite(save_dir + "frame%d.jpg" % count, crop_image)
                pbar.update(i)
                count += 1
    
        for file in tqdm(os.listdir(save_dir)):
            X.append(os.path.join(save_dir, file))
        
        return X

    def save_and_crop_4fps(self, video_path, crop_size):
        save_dir = video_path[:-4] + '_4fps/'
        self.frame_path = save_dir
        X = []

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            cap = cv2.VideoCapture(video_path)
            
            CURRENT_FPS = cap.get(cv2.CAP_PROP_FPS)
            transform_rate = int(CURRENT_FPS/4.0)
            assert transform_rate != 0

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 1
            pbar = tqdm(total = frame_count)

            count = 0
            while cap.isOpened():
                success, image = cap.read()
                if image is None:
                    print('Completed!')
                    break
                if count % transform_rate == 0:
                    crop_image = image[crop_size[1]:-crop_size[3],crop_size[0]:-crop_size[2]]
                    cv2.imwrite(save_dir + "frame%d.jpg" % int(count/transform_rate), crop_image)
                pbar.update(i)
                count += 1
    
        for file in tqdm(os.listdir(save_dir)):
            X.append(os.path.join(save_dir, file))
        
        return X

    def add_caption(self, model_name="xception", ckpt_path=""):
        save_dir = self.frame_path[:-1] + '_caption/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model = select_model(model=model_name, weights=None)
        model.load_weights(ckpt_path)

        results = classify(model, self)
        result_list = list(results)

        new_list = []
        for tuple in result_list:
            new_item = [int(os.path.basename(tuple[0])[5:-4]), tuple[1], tuple[2]]
            new_list.append(new_item)
        new_list.sort()

        assert len(new_list) == len(os.listdir(self.frame_path))
        
        for file in tqdm(os.listdir(self.frame_path)):
            result = new_list[int(file[5:-4])]
            image = cv2.imread(os.path.join(self.frame_path, file))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, result[2][0], (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(image, str(round(result[1][0]*100)/100), (500, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            
            cv2.putText(image, result[2][1], (50, 150), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(image, str(round(result[1][1]*100)/100), (500, 150), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            
            output_file = os.path.join(save_dir,file)
            cv2.imwrite(output_file, image)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        return tf.convert_to_tensor([load_image(fpath, (self.image_shape[0], self.image_shape[1]), self.scale) for fpath in batch_x])    

def create_video(frame_path ,save_dir):
    name = frame_path[-17:-14]
    num = len(os.listdir(frame_path))

    img = cv2.imread(os.path.join(frame_path,'frame0.jpg'))
    height, width, layers = img.shape
    size = (width, height)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    video = cv2.VideoWriter(save_dir + name +'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
    
    for i in tqdm(range(num)):
        img = cv2.imread(os.path.join(frame_path,'frame'+str(i)+'.jpg'))
        video.write(img)
    
    video.release()


if __name__ == "__main__":
    args = argparser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    image_shape = (299, 299, 3)
    model_name = "xception"

    for video in ['00870124_YOONSEONGKOO_11_38_27_249_001.avi']:
        video_path = '{0}/dataset/endoscopy/JFD-03K/videos/{1}'.format(
        os.getcwd()[:os.getcwd().find('/python/')], video)
        save_dir = '{0}/dataset/endoscopy/JFD-03K/videos/YSK/'.format(
        os.getcwd()[:os.getcwd().find('/python/')])

        dataset = EGDVideoDataset(video_path, save_dir, batch_size=args.batch, image_shape=image_shape, scale=args.scale) #,crop_size = (669, 1, 5, 1))

        # create_video(video_path[:-4]+'_4fps_caption/', video_path[:-4]+'_4fps_caption/')