import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# Load the pretrained Mask R-CNN model
#SegModel = models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
SegModel = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
# Define the transformations
trf = T.Compose([T.ToTensor()])

# get all image name
image_root = "MediaTek_Materials/MediaTek_Materials/yuv2png_rgb/"
image_name_list = os.listdir(image_root)

for image_name in image_name_list:
  # Load the image
  img_path = os.path.join(image_root, image_name)
  print("img_path",img_path)
  img = Image.open(img_path).convert("RGB")

  #inp = trf(img).unsqueeze(0).cuda()
  inp = trf(img).unsqueeze(0)
  # Predict the output
  with torch.no_grad():
      prediction = SegModel(inp)

  # Process the output
  pred_score = prediction[0]['scores'].detach().cpu().numpy()
  pred_boxes = prediction[0]['boxes'].detach().cpu().numpy()
  pred_labels = prediction[0]['labels'].detach().cpu().numpy()
  pred_masks = prediction[0]['masks'].detach().cpu().numpy()

  # Set a threshold to filter out low confidence detections
  threshold = 0.5
  pred_t = pred_score >= threshold

  pred_boxes = pred_boxes[pred_t]
  pred_labels = pred_labels[pred_t]
  pred_masks = pred_masks[pred_t]

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

  # Visualize the results
  def plot_instance_segmentation(image, boxes, masks, labels, classes):
      # Convert the image to an RGB image
      image = np.array(image)

      # Generate random colors
      colors = np.random.randint(0, 255, (len(boxes), 3), dtype=np.uint8)

      for i in range(len(boxes)):
          box = boxes[i]
          mask = masks[i, 0]
          label = labels[i]
          class_name = classes[label]
          color = colors[i]

          # Draw bounding box
          image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color.tolist(), 2)

          # Add class name
          font = cv2.FONT_HERSHEY_SIMPLEX
          image = cv2.putText(image, class_name, (int(box[0]), int(box[1])-10), font, 0.5, color.tolist(), 2, cv2.LINE_AA)

          # Apply mask to the image
          mask = (mask > 0.5).astype(np.uint8)
          colored_mask = np.zeros_like(image, dtype=np.uint8)
          label_mask = np.zeros_like(image, dtype=np.uint8)
          for c in range(3):
              colored_mask[:, :, c] = mask * color[c]
              label_mask[:,:,c] = mask * label
          image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
          label_mask_path = os.path.join(image_root, image_name.replace(".png","_label_mask.npy"))
          np.save(label_mask_path, label_mask)


      return Image.fromarray(image)

  segmented_img = plot_instance_segmentation(img.copy(), pred_boxes, pred_masks, pred_labels, COCO_INSTANCE_CATEGORY_NAMES)

  # Overlay the segmentation results on the original image
  combined_img = Image.blend(img, segmented_img, alpha=0.5)

  # Display the combined image
  plt.figure(figsize=(12, 12))
  plt.imshow(combined_img)
  plt.axis('off')
  plt.show()

  # Save the combined image
  combined_image_path = os.path.join(image_root, image_name.replace(".png","_segmented_combined.png"))
  # combined_img.save('/content/drive/MyDrive/cv_Final/MediaTek_Materials/MediaTek_Materials/yuv2png_rgb/000_segmented.png')
  combined_img.save(combined_image_path)
