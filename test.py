import cv2
import time

from utils.tool import *
from module.detector import Detector

if __name__ == '__main__':
    # 指定训练配置文件
    

    yaml ='./data/firuts.yaml'
    weight ='./checkpoint/weight_AP05_0.980960_70-epoch.pth'
    onnx = True
    #opt = parser.parse_args()
    cpu = True
    thresh = 0.8

    # 1 加载视频文件
    capture = cv2.VideoCapture(0)
    fps_time = time.time()
    #ret,frame = cap.read()
    device = torch.device("cpu")

    # 配置yaml配置文件
    cfg = LoadYaml(yaml)   
    print(cfg)

    # 模型加载
    print("load weight from:%s"%weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(weight))
    #sets the module in eval node
    model.eval()


    while True:
        ret, frame = capture.read()
    # 数据预处理
        ori_img =cv2.flip(frame,1)
        #ori_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res_img = cv2.resize(ori_img, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

    # 导出onnx模型
    #     if onnx:
    #         torch.onnx.export(
    #                         model,                    #model being run
    #                         img,                 # model input (or a tuple for multiple inputs)
    #                         "./FastestDet.onnx",             # where to save the model (can be a file or file-like object)
    #                         export_params=True,        # store the trained parameter weights inside the model file
    #                         opset_version=11,          # the ONNX version to export the model to
    #                         do_constant_folding=True)  # whether to execute constant folding for optimization

        # 模型推理
        #start = time.perf_counter()
        preds = model(img)
        #end = time.perf_counter()
        #time = (end - start) * 1000.
        #print("forward time:%fms"%time)

        # 特征图后处理
        output = handel_preds(preds, device, thresh)

        # 加载label names
        LABEL_NAMES = []
        with open(cfg.names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        H, W, _ = ori_img.shape
        scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

        # 绘制预测框
        for box in output[0]:
            print(box)
            box = box.tolist()
        
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        now = time.time()
        fps_text = 1/(now - fps_time)
        fps_time =  now
        cv2.putText(ori_img,str(round(fps_text,2)),(20,20),cv2.FONT_ITALIC,1,(0,255,0),2)
        cv2.imshow("demo",ori_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 5 释放资源
    capture.release()
    cv2.destroyAllWindows()
        #cv2.imwrite("result.png", ori_img)
