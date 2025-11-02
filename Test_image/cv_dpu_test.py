import os
import argparse
import time
import numpy as np
import cv2
import random
import colorsys
from pynq_dpu import DpuOverlay

# ------------- Helper functions (adapted from your original) -------------
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = [c.strip() for c in f.readlines()]
    return class_names

def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def pre_process(image, model_image_size):
    # convert BGR->RGB and pad/resize
    image = image[..., ::-1]
    image_h, image_w, _ = image.shape
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0 and model_image_size[1] % 32 == 0
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32') / 255.0
    image_data = np.expand_dims(image_data, 0)
    return image_data

def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1).astype(np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype=np.float32)
    image_shape = np.array(image_shape, dtype=np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores

def nms_boxes(boxes, scores, iou_threshold=0.55):
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def evaluate(yolo_outputs, image_shape, class_names, anchors, score_thresh=0.2):
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]
    boxes = []
    box_scores = []
    input_shape = np.array(np.shape(yolo_outputs[0])[1:3]) * 32
    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes); box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    mask = box_scores >= score_thresh
    boxes_, scores_, classes_ = [], [], []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        if class_boxes_np.size == 0:
            continue
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np)
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype=np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    if not boxes_:
        return np.zeros((0,4), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)
    return boxes_, scores_, classes_

def draw_and_save(image, boxes, scores, classes_idx, class_names, colors, out_path):
    # draw using OpenCV and save to out_path
    im = image.copy()
    h, w = im.shape[:2]
    for i, box in enumerate(boxes):
        # box format from evaluate is [x_min,y_min,x_max,y_max]
        left, top, right, bottom = [int(x) for x in box]
        score = float(scores[i])
        cls = int(classes_idx[i])
        color = colors[cls]
        cv2.rectangle(im, (left, top), (right, bottom), color, max(1, int(0.002 * (w + h))))
        label = f"{class_names[cls]}:{score:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        cv2.rectangle(im, (left, top - t_size[1] - 4), (left + t_size[0] + 4, top), color, -1)
        cv2.putText(im, label, (left + 2, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1)
    cv2.imwrite(out_path, im)

# ------------- main program -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to xmodel (or xclbin as used by your overlay loader)')
    parser.add_argument('--images', required=True, help='Folder with input images (JPEG/JPG/PNG)')
    parser.add_argument('--classes', required=True, help='Path to class names txt')
    parser.add_argument('--output', required=True, help='Folder to save annotated images')
    parser.add_argument('--score-thresh', type=float, default=0.20)
    args = parser.parse_args()

    # anchors for your model (from original)
    anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
    anchors = np.array([float(x) for x in anchor_list]).reshape(-1,2)

    class_names = get_class(args.classes)
    num_classes = len(class_names)

    # colors
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = [ (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors ]
    random.seed(0); random.shuffle(colors); random.seed(None)

    # prepare output folder
    os.makedirs(args.output, exist_ok=True)

    # load overlay and model
    print("Loading overlay/model:", args.model)
    overlay = DpuOverlay()
    overlay.load_model(args.model)
    dpu = overlay.runner
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut0 = tuple(outputTensors[0].dims)
    shapeOut1 = tuple(outputTensors[1].dims)
    shapeOut2 = tuple(outputTensors[2].dims)
    outputSize0 = int(outputTensors[0].get_data_size() / shapeIn[0])
    outputSize1 = int(outputTensors[1].get_data_size() / shapeIn[0])
    outputSize2 = int(outputTensors[2].get_data_size() / shapeIn[0])

    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    output_data = [
        np.empty(shapeOut0, dtype=np.float32, order="C"),
        np.empty(shapeOut1, dtype=np.float32, order="C"),
        np.empty(shapeOut2, dtype=np.float32, order="C"),
    ]
    image_blob = input_data[0]

    # list images
    images = [f for f in os.listdir(args.images) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    images.sort()
    print(f"Found {len(images)} images in {args.images}")

    for idx, img_name in enumerate(images):
        in_path = os.path.join(args.images, img_name)
        out_path = os.path.join(args.output, f"annot_{img_name}")
        print(f"[{idx+1}/{len(images)}] Processing {img_name} -> {out_path}")
        img = cv2.imread(in_path)
        if img is None:
            print("  ERROR: failed to read", in_path)
            continue
        H, W = img.shape[:2]
        # preprocess
        image_data = pre_process(img, (416, 416)).astype(np.float32)
        image_blob[0,...] = image_data.reshape(shapeIn[1:])
        # run DPU
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        # collect outputs and reshape
        conv_out0 = np.reshape(output_data[0], shapeOut0)
        conv_out1 = np.reshape(output_data[1], shapeOut1)
        conv_out2 = np.reshape(output_data[2], shapeOut2)
        yolo_outputs = [conv_out0, conv_out1, conv_out2]
        # decode
        boxes, scores, classes_idx = evaluate(yolo_outputs, (H, W), class_names, anchors, score_thresh=args.score_thresh)
        # draw and save
        draw_and_save(img, boxes, scores, classes_idx, class_names, colors, out_path)
        print(f"  Saved {out_path}  Detected: {len(boxes)}")
    # cleanup
    del overlay
    del dpu
    print("Done processing all images.")

if __name__ == '__main__':
    main()
