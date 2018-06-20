import cv2
import os
import numpy as np


def debug_annotations(annotations):
    with open(annotations, "r") as file:
        for line in file:
            if line.strip() == "":
                continue
            line = line.strip().split(",")
            path, x1, y1, x2, y2, label = line[:6]
            if not x1.strip() == "":
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                print("(" + str(x1) + "," + str(y1) + ")-(" + str(x2) + "," + str(y2) + ")")
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            print(image.shape)

            width = 800
            scale = float(width) / float(image.shape[1])
            height = int(float(image.shape[0]) * scale)

            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            x1 = min(width - 1, max(0, x1))
            x2 = min(width - 1, max(0, x2))
            y1 = min(height - 1, max(0, y1))
            y2 = min(height - 1, max(0, y2))
            image = cv2.resize(image, (width, height))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(image, label, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("annotation", image)
            cv2.waitKey(1)


def mark_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if len(pixels) >= 4:
            return
        if x < 7:
            x = 0
        elif x > original_dims[0] - 7:
            x = original_dims[0]
        if y < 7:
            y = 0
        elif y > original_dims[1] - 7:
            y = original_dims[1]
        global marked_pixel
        marked_pixel = (x, y)
        pixels.append(marked_pixel)


def labeling(input_dir, annotations):
    global pixels
    pixels = []
    annotations_file_content = []
    images_to_ignore = []
    with open(annotations, "r") as annotations_file:
        for line in annotations_file:
            annotations_file_content.append(line)
            line_split = line.split(",")
            images_to_ignore.append(line_split[0])
    class_names = []
    bb_min = (-1, -1)
    bb_max = (-1, -1)
    global original_dims
    original_dims = (0, 0)
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
                                cv2.rectangle(image, bb_min, bb_max, (0, 255, 0))
                                cv2.putText(image, class_name, (bb_min[0], bb_min[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 1, cv2.LINE_AA)
                            else:
                                cv2.circle(image, pixel, 3, (0, 255, 0), -1)
                        cv2.imshow(class_name, image)
                        key = cv2.waitKey(1) & 0xFF

                        if key == 27 and len(pixels) > 0:
                            pixels.pop()

                        # if the 'c' key is pressed, break from the loop
                        elif key == ord("c"):
                            quit(0)
                        elif key == ord("n"):
                            annotations_file.write(img_path + ",")
                            annotations_file.write(",")
                            annotations_file.write(",")
                            annotations_file.write(",")
                            annotations_file.write(",")
                            annotations_file.write("\n")
                            print("saved annotations for image: " + img_path)
                            pixels = []
                            break
                        elif key == 13:
                            if len(pixels) == 4:
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
                                if bb[0] > original_dims[0] - 12:
                                    bb[0] = original_dims[0]
                                if bb[2] > original_dims[0] - 12:
                                    bb[2] = original_dims[0]
                                if bb[1] > original_dims[1] - 12:
                                    bb[1] = original_dims[1]
                                if bb[3] > original_dims[1] - 12:
                                    bb[3] = original_dims[1]
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


if __name__ == '__main__':
    input_dir = "D:/Documents/Villeroy & Boch - Subway 2.0"
    annotations = "D:/Documents/Villeroy & Boch - Subway 2.0/annotations.csv"
    labeling(input_dir, annotations)
    #debug_annotations(annotations)
