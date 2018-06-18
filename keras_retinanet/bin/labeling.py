import cv2
import os
import numpy as np

pixels = []


def mark_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if len(pixels) >= 4:
            return
        global marked_pixel
        marked_pixel = (x, y)
        pixels.append(marked_pixel)


if __name__ == '__main__':
    input_dir = "D:/Documents/Villeroy & Boch - Subway 2.0"
    annotations = "D:/Documents/Villeroy & Boch - Subway 2.0/annotations.csv"
    annotations_file_content = []
    images_to_ignore = []
    with open(annotations) as annotations_file:
        for line in annotations_file:
            annotations_file_content.append(line)
            line_split = line.split(",")
            images_to_ignore.append(line_split[0])
    class_names = []
    bb_min = (-1, -1)
    bb_max = (-1, -1)
    original_dims = (0,0)
    with open(annotations, "w") as annotations_file:
        for line in annotations_file_content:
            annotations_file.write(line)
        for dirpath, subdirs, files in os.walk(input_dir):
            for x in files:
                if x.endswith(".jpg") or x.endswith(".JPG") or x.endswith(".png") or x.endswith(".PNG"):
                    class_name = os.path.basename(dirpath)
                    img_path = os.path.join(dirpath, x)
                    if img_path in images_to_ignore:
                        continue
                    if not class_name in class_names:
                        class_names.append(class_name)
                    cv2.namedWindow(class_name)
                    cv2.setMouseCallback(class_name, mark_pixel)
                    original_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    original_dims = (original_image.shape[1], original_image.shape[0])
                    image = original_image.copy()
                    width = 1000
                    aspect = width / image.shape[1]
                    height = int(aspect * image.shape[0])
                    image = cv2.resize(image, dsize=(width, height))
                    clone = image.copy()
                    while True:
                        # display the image and wait for a keypress
                        for pixel in pixels:
                            if len(pixels) == 4:
                                bb_min = (min(np.array(pixels)[:, :1])[0], min(np.array(pixels)[:, 1:])[0])
                                bb_max = (max(np.array(pixels)[:, :1])[0], max(np.array(pixels)[:, 1:])[0])
                                cv2.rectangle(image, bb_min, bb_max, (0,255,0))
                            else:
                                cv2.circle(image, pixel, 3, (0, 255, 0), -1)
                        cv2.imshow("image", image)
                        key = cv2.waitKey(1) & 0xFF

                        if key == 27 and len(pixels) > 0:
                            pixels.pop()

                        # if the 'c' key is pressed, break from the loop
                        elif key == ord("c"):
                            quit(0)
                        elif key == 13:
                            if len(pixels) == 4:
                                restore_aspect = 1.0 - aspect
                                bb = np.array([bb_min[0], bb_min[1], bb_max[0], bb_max[1]], np.float32)
                                bb[0] = bb[0] / width * original_dims[0]
                                bb[1] = bb[1] / height * original_dims[1]
                                bb[2] = bb[2] / width * original_dims[0]
                                bb[3] = bb[3] / height * original_dims[1]
                                bb = bb.astype(np.int16)
                                if bb[0] > bb[2]:
                                    tmp = bb[0]
                                    bb[0] = bb[2]
                                    bb[2] = tmp
                                if bb[1] > bb[3]:
                                    tmp = bb[1]
                                    bb[1] = bb[3]
                                    bb[3] = tmp
                                annotations_file.write(img_path + ",")
                                annotations_file.write(str(bb[0]) + ",")
                                annotations_file.write(str(bb[1]) + ",")
                                annotations_file.write(str(bb[2]) + ",")
                                annotations_file.write(str(bb[3]) + ",")
                                annotations_file.write(class_name + "\n")
                                print("saved annotations for image: " + img_path)
                                pixels = []
                                break

                        image = clone.copy()
