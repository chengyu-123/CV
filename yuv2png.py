import os, sys, argparse
import numpy as np
from PIL import Image

def convert(yuv_file, output_dir):
    f_y = open(yuv_file, "rb")
    w ,h = 3840, 2160
    seq_len = 129
    frame_size = int(3/2 * w * h)
    for frame_num in range(seq_len):
        converted_image = Image.new('L', (w, h))
        pixels = converted_image.load()

        f_y.seek(frame_size * frame_num)
        
        for i in range(h):
            for j in range(w):
                y = ord(f_y.read(1))
                pixels[j,i] = int(y)

        converted_image.save(os.path.join(output_dir, '%03d.png' % frame_num), "png")

    f_y.close()

def convert_rgb(yuv_file, output_dir):
    f_y = open(yuv_file, "rb")
    w, h = 3840, 2160
    seq_len = 129
    frame_size = int(3 / 2 * w * h)
    
    for frame_num in range(seq_len):
        # Create arrays for Y, U, and V planes
        y_plane = np.zeros((h, w), dtype=np.uint8)
        u_plane = np.zeros((h // 2, w // 2), dtype=np.uint8)
        v_plane = np.zeros((h // 2, w // 2), dtype=np.uint8)

        # Read Y plane
        f_y.seek(frame_size * frame_num)
        y_plane = np.frombuffer(f_y.read(w * h), dtype=np.uint8).reshape((h, w))
        
        # Read U plane
        u_plane = np.frombuffer(f_y.read(w * h // 4), dtype=np.uint8).reshape((h // 2, w // 2))
        
        # Read V plane
        v_plane = np.frombuffer(f_y.read(w * h // 4), dtype=np.uint8).reshape((h // 2, w // 2))

        # Upsample U and V planes
        u_plane = u_plane.repeat(2, axis=0).repeat(2, axis=1)
        v_plane = v_plane.repeat(2, axis=0).repeat(2, axis=1)

        # Stack Y, U, and V planes to form a YUV image
        yuv = np.stack((y_plane, u_plane, v_plane), axis=-1)

        # Convert YUV to RGB
        rgb = yuv2rgb(yuv)

        # Convert numpy array to PIL image
        converted_image = Image.fromarray(rgb, 'RGB')

        # Save the image
        converted_image.save(os.path.join(output_dir, '%03d.png' % frame_num), "PNG")

    f_y.close()

def yuv2rgb(yuv):
    # Constants for YUV to RGB conversion
    m = np.array([[1.164,  0.0,  1.596],
                  [1.164, -0.392, -0.813],
                  [1.164,  2.017,  0.0]])
    
    yuv = yuv.astype(np.float32)
    yuv[:, :, 0] -= 16
    yuv[:, :, 1:] -= 128
    yuv = np.dot(yuv, m.T)
    np.clip(yuv, 0, 255, out=yuv)
    return yuv.astype(np.uint8)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yuv_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()

    yuv_file, output_dir = args.yuv_file, args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # convert(yuv_file, output_dir)
    convert_rgb(yuv_file, output_dir)
