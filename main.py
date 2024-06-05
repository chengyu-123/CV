import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# Load the pretrained Mask R-CNN model for instance segmentation
SegModel = models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()

# Load the pretrained DeepLabV3 model for semantic segmentation
SegModel_deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).cuda().eval()

image_root = '/MediaTek_Materials/MediaTek_Materials/yuv2png_rgb'
image_list = os.listdir(image_root)

for image_name in image_list:
  # Load the image
  img_path = os.path.join(image_root, image_name)
  img = Image.open(img_path).convert("RGB")

  # Define the transformations
  trf = T.Compose([T.ToTensor()])
  inp = trf(img).unsqueeze(0).cuda()

  # Predict the output for instance segmentation
  with torch.no_grad():
      prediction = SegModel(inp)

  # Process the output for instance segmentation
  pred_score  = prediction[0]['scores'].detach().cpu().numpy()
  pred_boxes  = prediction[0]['boxes'].detach().cpu().numpy()
  pred_labels = prediction[0]['labels'].detach().cpu().numpy()
  pred_masks  = prediction[0]['masks'].detach().cpu().numpy()

  # Set a threshold to filter out low confidence detections
  threshold = 0.5
  pred_t = pred_score >= threshold

  pred_boxes = pred_boxes[pred_t]
  pred_labels = pred_labels[pred_t]
  pred_masks = pred_masks[pred_t]

  # Predict the output for semantic segmentation
  with torch.no_grad():
      output = SegModel_deeplab(inp)['out'][0]
  output_predictions = output.argmax(0).byte().cpu().numpy()

  # Define COCO classes
  COCO_INSTANCE_CATEGORY_NAMES = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
      'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
      'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
      'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
      'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
      'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
      'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
      'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
      'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
      'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
  ]

  # Define a function to visualize instance segmentation results
  def plot_instance_and_semantic_segmentation(image, boxes, masks, labels, classes, semantic_output_predictions, road_class=0, sky_class=21):
      image = np.array(image)
      colors = np.random.randint(0, 255, (len(boxes), 3), dtype=np.uint8)
      seg_map = np.zeros_like(image, dtype=np.uint8)
      for i in range(len(boxes)):
          box = boxes[i]
          mask = masks[i, 0]
          label = labels[i]
          class_name = classes[label]
          color = colors[i]
          image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color.tolist(), 2)
          font = cv2.FONT_HERSHEY_SIMPLEX
          image = cv2.putText(image, class_name, (int(box[0]), int(box[1])-10), font, 0.5, color.tolist(), 2, cv2.LINE_AA)
          mask = (mask > 0.5).astype(np.uint8)
          colored_mask = np.zeros_like(image, dtype=np.uint8)
          for c in range(3):
              colored_mask[:, :, c] = mask * color[c]
              seg_map[:, :, c] = mask * label
          image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

      # sematic segmentation
      road_mask = output_predictions == road_class
      sky_mask = output_predictions == sky_class
      instant_mask = masks
      instant_mask_bin = ((instant_mask > 0.5))
      instant_mask_bin = (instant_mask_bin.sum(0)> 0.5)[0]

      # image = np.array(image)
      road_seg_mask = np.zeros_like(image, dtype=np.uint8)
      # print('colored_mask',colored_mask.shape)
      # stop()
      road_seg_mask[road_mask] = [255, 0, 0]  # Color the road area red
      # road_seg_mask[sky_mask] = [0, 0, 255]   # Color the sky area blue
      road_seg_mask[instant_mask_bin] = [0, 0, 0]  # Color the instant area black
      road_mask[:1300] = False
      road_mask[instant_mask_bin] = False
      road_seg_mask[:1300] = [0, 0, 0]
      for c in range(3):
        seg_map[:, :, c] = road_mask * 999 # define road is label 127

      # colored_mask = colored_mask + instant_mask
      image = cv2.addWeighted(image, 1, road_seg_mask, 0.5, 0)
      seg_result_label_path = os.path.join(image_root, image_name.replace(".png","_label.npy"))

      np.save(seg_result_label_path, seg_map)

      return Image.fromarray(image)

  # Plot instance segmentation results
  segmented_img = plot_instance_and_semantic_segmentation(img.copy(), pred_boxes, pred_masks, pred_labels, COCO_INSTANCE_CATEGORY_NAMES, output_predictions,)

  # Plot semantic segmentation results
  # semantic_segmented_img = plot_semantic_segmentation(img.copy(), output_predictions, pred_masks)

  # Overlay the segmentation results on the original image
  combined_img = Image.blend(img, segmented_img, alpha=0.5)
  # combined_img = Image.blend(combined_img, semantic_segmented_img, alpha=0.5)

  # Display the combined image
  plt.figure(figsize=(12, 12))
  plt.imshow(combined_img)
  plt.axis('off')
  plt.show()

  # Save the combined image
  seg_result_img_path = os.path.join(image_root, image_name.replace(".png","_segmented.png"))
  combined_img.save(seg_result_img_path)
  # stop()

