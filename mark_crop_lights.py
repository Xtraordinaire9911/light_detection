'''Auto mark, crop, save ground lights from runway images.'''

import numpy as np
import pandas as pd

import platform
from platform import subprocess
import os
import cv2
from matplotlib import pyplot as plt
from copy import copy


class Mark_Crop:
    def __init__(self, camera, image):
        # global l_camera, dir_curr, dir_all_images
        self.dir_curr = os.path.join(dir_all_images, l_camera[camera])
        print('dir_curr:', self.dir_curr)
        self.pos = []

        self.camera = sorted(l_camera)[camera]
        self.image_name = sorted(os.listdir(self.dir_curr))[image]
        # print('self.image_name', self.image_name)
        print('camera:', self.camera, ' image:', 'No.' + str(image), '  image_name:', self.image_name)
        self.img = cv2.imread(os.path.join(self.dir_curr, self.image_name))

        # print(self.img)
        #   cv2.imshow('Test', self.img)

        self.img_hover = self.img.copy()
        self.img_click = self.img.copy()
        self.pressedkey = cv2.waitKey(0)

    def save_data(self):
        # Wait for ESC key to exit

        # subprocess.run(['xdg-open', '1593612224.2866488.jpg'])
        #   cv2.imshow('Test', self.img)
        # print(self.img)
        # if self.pressedkey == 27:
        print('positions of runway lights:', self.pos)
        d_pos_lights[self.camera + '/' + self.image_name] = self.pos + [np.nan for i in range(10 - len(self.pos))]
        df_pos_lights = pd.DataFrame(d_pos_lights).T
        df_pos_lights.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if not os.path.exists(self.dir_curr + '/crop/log_pos_lights_{}.csv'.format(self.camera)):
            df_pos_lights.to_csv(self.dir_curr + '/crop/log_pos_lights_{}.csv'.format(self.camera))
        else:
            df_pos_lights[-1:].to_csv(self.dir_curr + '/crop/log_pos_lights_{}.csv'.format(self.camera), mode = 'a', header = False)

        df = pd.read_csv(self.dir_curr + '/crop/log_pos_lights_{}.csv'.format(self.camera))
        if df.shape[0] % 100 == 0:
            '''for recover use'''
            df.to_csv(self.dir_curr + '/crop/log_pos_lights_{}(recover:{}).csv'.format(self.camera, str(df.shape[0])))
        # print('d_pos_lights:', d_pos_lights)
        del self.img, self.img_hover, self.img_click
        del df, df_pos_lights
        cv2.destroyAllWindows()

    def draw_square(self, event, x, y, flags, param):
        # print('ooooooooooooooooo')
        global num_clicks
        mouseX, mouseY = copy(x), copy(y)
        # print('x_now:', mouseX, ' y_now:', mouseY)
        # print('height', self.img.shape[1], 'width', self.img.shape[0])
        cv2.circle(self.img_click, (x, y), 8, (255, 0, 0), -1)
        cv2.rectangle(self.img_click, (max(mouseX - 128, 0), max(mouseY - 128, 0)), (min(mouseX + 128, self.img.shape[1]), min(mouseY + 128, self.img.shape[0])), (0, 255, 0), 15)
        cv2.imshow('image viewer - ' + self.image_name, self.img_click)
        self.img_click = self.img_hover.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            num_clicks += 1
            print('num_clicks:', num_clicks)
            # print('pos_light:', (mouseX, mouseY))

            if len(self.pos) == 0:
                self.pos.append((mouseX, mouseY))
                # print('mouseX:', mouseX, 'mouseY:', mouseY)
            elif (mouseX, mouseY) != self.pos[-1]:
                self.pos.append((mouseX, mouseY))
                # print('mouseX:', mouseX, 'mouseY:', mouseY)
                # print(pos)

            cv2.circle(self.img_hover, (mouseX, mouseY), 10, (255, 0, 0), -1)
            cv2.rectangle(self.img_hover, (max(mouseX - 128, 0), max(mouseY - 128, 0)), (min(mouseX + 128, self.img.shape[1]), min(mouseY + 128, self.img.shape[0])), (0, 0, 255), 15)
            self.img_click = self.img_hover.copy()
            img_crop = self.img[max(mouseY - 128, 0):min(mouseY + 128, self.img.shape[0]), max(mouseX - 128, 0):min(mouseX + 128, self.img.shape[1])]
            cv2.imwrite(os.path.join(dir_all_images, self.camera) + '/crop' + '/{}_crop_{}'.format(
                self.image_name.strip('.jpg'),
                str(num_clicks) + '.jpg'),
                        img_crop)
            print('cropped image:', os.path.join(dir_all_images, self.camera) + '/crop' + '/{}_crop_{}'.format(
                self.image_name.strip('.jpg'),
                str(num_clicks) + '.jpg'))
            cv2.imshow('image viewer - ' + self.image_name, self.img_hover)


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------'''


dir_all_images = '/home/flyinstinct/PycharmProjects/images'    # "/home/flyinstinct/PycharmProjects/light_detection/images",
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/images'
l_camera = sorted([img for img in os.listdir(dir_all_images)])[1:]
print('l_camera', l_camera)
d_num_img_camera = {l_camera[i]: len([img for img in os.listdir(os.path.join(dir_all_images, l_camera[i])) if img.endswith('.jpg')])\
                    for i in range(len(l_camera))}
print('num_img_camera', d_num_img_camera)


num_clicks = 0
print('num_clicks', num_clicks)
# mouseX = -1
# mouseY = -1
# pos = []
d_pos_lights = {}
# df = pd.DataFrame(columns = ['img', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def mark_and_crop(i, j):
    '''i-th camera; j-th image of the camera; 0-indexed'''

    # user_platform = platform.system()
    # # print('userplatform', user_platform)
    # if user_platform == 'Linux':

    image_loader = Mark_Crop(i, j)
    # print('ok')
    if image_loader.image_name.endswith('.jpg'):
        # print('yes')
        # count = 0


        cv2.namedWindow('image viewer - ' + image_loader.image_name, cv2.WINDOW_GUI_EXPANDED)
        # print('fine')

        cv2.resizeWindow('image viewer - ' + image_loader.image_name, 1200, 1200)
        cv2.setMouseCallback('image viewer - ' + image_loader.image_name, image_loader.draw_square)
        # print('111111111')
        # print('count', count)
        # while True:
        # count += 1
        # print('count', count)
        # print('looooooooooping')
        key = cv2.waitKey(0) & 0xFF
        image_loader.save_data()
        # if key == ord('n'):
        #     break
        # cv2.imshow('image viewer', image_loader.img)
        # plt.imshow(self.img, 'image')
        # subprocess.run(['xdg-open', '1593612224.2866488.jpg'])
        # subprocess.call(['xdg-open', os.path.join(dir_curr, image_loader.image_name)])

        # print('222222222222222222')

        # print('count', count)


def main():
    global  num_clicks
    '''i-th camera; j-th image of the camera; 0-indexed'''
    for i in range(3, 4):  # len(l_camera)
        print('************************** NOW  WORKING  ON  THE  CAMERA No.' + str(i) + ' ... ... **************************')
        if cv2.waitKey(0) == 27:
            break
        for j in range(d_num_img_camera[l_camera[i]]):
            print('-- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * --')
            print('now processing the image No.' + str(j) + ' ... ...')
            num_clicks = 0
            # if cv2.waitKey(0) == 27:
            #     break
            # elif cv2.waitKey(0) == ord('n'):
            #     continue
            mark_and_crop(i, j)
        print('!!!!!!!!!! COMPLETED WORKING ON CAMERA No.{} !!!!!!!!!!'.format(str(i)))
    print('!!!!!!!!!!!!!!!  JOB FINISHED  !!!!!!!!!!!!!!!!')

if __name__ == '__main__':
    main()