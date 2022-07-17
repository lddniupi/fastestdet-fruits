import os
import cv2
import time
import argparse

import torch
import model.detector
import utils.utils
import utils.datasets

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.data', help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights/coco-290-epoch-0.906122ap-model.pth',
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='img/017.jpg', help='The path of test image')
    parser.add_argument('--vid', type=str, default='person.mp4', help='The path of test image')
    parser.add_argument('--view_img', type=bool, default=True, help='visual the vedio')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    # assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    # 1 加载视频文件
    capture = cv2.VideoCapture(0)
    # 2 读取视频
    ret, frame = capture.read()
    while ret:
        # 3 ret 是否读取到了帧，读取到了则为True
        cv2.imshow("video", frame)
        ret, frame = capture.read()

        # 数据预处理
        img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 模型推理
        preds = model(img)
        # 特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.4, iou_thres=0.4)

        # 加载label names
        LABEL_NAMES = []
        with open(cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())

        h, w, _ = frame.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        # 绘制预测框
        for box in output_boxes[0]:
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        # 4 若键盘按下q则退出播放
        if cv2.waitKey(20) & 0xff == ord('q'):
            break

    # 5 释放资源
    capture.release()
    # 6 关闭所有窗口
    cv2.destroyAllWindows()