import json
import os
import colorsys
from nets.slowfastnet import slowfastnet as Network
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

map_mode            = 0

cocoGt_path         = '/home/public/IRDST/real/IRDST1_anns/instances_test2017.json'
dataset_img_path    = '/home/public/IRDST/real/'

temp_save_path      = 'map_out/coco_eval'

class MAP_vid(object):
    _defaults = {
        "model_path"        : '/home/dww/OD/CoMoE-bei/logs/loss_2025_11_26_11_37_57/ep024-loss1.062-val_loss2.279.pth',
        "classes_path"      : 'model_data/classes.txt',
        "input_shape"       : [512, 512],
        "phi"               : 's',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes  = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net    = Network(self.num_classes, num_frame=5)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image_id, images,domain_id, results):
        image_shape = np.array(np.shape(images[0])[0:2])
        images       = [cvtColor(image) for image in images]
        image_datas  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        image_datas  = [np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0) for image_data in image_datas]
        images       = np.concatenate(image_datas, 0)

        images       = torch.from_numpy(images).permute(1,0,2,3).unsqueeze(0)

        with torch.no_grad():
            if self.cuda:
                images = images.cuda()

            domain_ids = torch.tensor([domain_id])
            if self.cuda:
                domain_ids = domain_ids.cuda()
            outputs, _ = self.net(images, domain_ids=domain_ids)
            outputs = decode_outputs(outputs, self.input_shape)
            outputs_nms = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

            if outputs_nms[0] is None: 
                return results

            top_label   = np.array(outputs_nms[0][:, 6], dtype = 'int32')
            top_conf    = outputs_nms[0][:, 4] * outputs_nms[0][:, 5]
            top_boxes   = outputs_nms[0][:, :4]

        for i, c in enumerate(top_label):
            result = {}
            top, left, bottom, right = [float(x) for x in top_boxes[i]]
            result["image_id"]       = int(image_id)
            result["category_id"]    = int(clsid2catid[int(c)])
            result["bbox"]           = [left, top, right - left, bottom - top]
            result["score"]          = float(top_conf[i])
            results.append(result)

        return results

def get_history_imgs(line):
    dir_path = line.replace(line.split('/')[-1], '')
    file_type = line.split('.')[-1]
    index = int(line.split('/')[-1].split('.')[0])

    return [os.path.join(dir_path, "%d.%s" % (max(id, 0), file_type)) for id in range(index - 4, index + 1)]

def get_domain_id(file_path):
    path_parts = file_path.split('/')
    for part in path_parts:
        if 'DAUB-H' in part:
            return 1
        elif 'IRDST' in part:
            return 2
        elif 'ITSDT' in part:
            return 3
    return 0

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = cocoGt.getImgIds()
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = MAP_vid(confidence = 0.001, nms_iou = 0.65)

        results = []
        for image_id in tqdm(ids):
            image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
            image_paths = get_history_imgs(image_path)
            images = [Image.open(item) for item in image_paths]
            domain_id = get_domain_id(image_path)
            results = yolo.detect_image(image_id, images, domain_id, results)

        print(f"Total detections: {len(results)}")
        with open(os.path.join(temp_save_path, 'coco_results.json'), "w") as f:
            json.dump(results, f, cls=NumpyEncoder)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'coco_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get map done.")

        precisions = cocoEval.eval['precision']
        precision_50 = precisions[0,:,0,0,-1]
        recalls = cocoEval.eval['recall']
        recall_50 = recalls[0,0,0,-1]

        print("Precision: %.4f, Recall: %.4f, F1: %.4f" %(np.mean(precision_50[:int(recall_50*100)]), recall_50, 2*recall_50*np.mean(precision_50[:int(recall_50*100)])/( recall_50+np.mean(precision_50[:int(recall_50*100)]))))
        print("Get map done.")
