import cv2
import os
import numpy as np
import time
import csv


def debug_annotations(annotations, dir="", max_width=None):
    image_data = {}
    with open(annotations, "r") as file:
        for line in file:
            if line.strip() == "":
                continue
            line = line.strip().split(",")
            path, x1, y1, x2, y2, label = line[:6]
            if not x1.strip() == "":
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                if path not in image_data:
                    image_data[path] = [(x1, y1, x2, y2, label)]
                else:
                    image_data[path].append((x1, y1, x2, y2, label))
            else:
                image_data[path] = []
        for i, key in enumerate(image_data):
            image = cv2.imread(os.path.join(dir, key.replace("\"", "")), cv2.IMREAD_COLOR)
            print(len(image_data[key]))
            waitKey = 1
            for box in image_data[key]:
                x1, y1, x2, y2, label = box[0], box[1], box[2], box[3], box[4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if (x1,y1,x2,y2) == (51,345,53,345):
                    waitKey = 0
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(image, label, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if max_width:
                width = max_width
                scale = float(width) / float(image.shape[1])
                height = int(float(image.shape[0]) * scale)
            else:
                width = image.shape[1]
                height = image.shape[0]
            image = cv2.resize(image, (width, height))
            if waitKey == 0:
                cv2.imshow("annotation", image)
                cv2.waitKey(waitKey)


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


def remove_missing(file):
    new_annotations = []
    with open(file, "r") as f:
        for line in f:
            image = line.split(",")[0]
            if os.path.isfile(image):
                new_annotations.append(line)
    if len(new_annotations) > 0:
        with open(file, "w") as f:
            for line in new_annotations:
                f.write(line)


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


def parse_annotations_from_alpha_mask(dir, separator, background_color1, background_color2, mode="retinanet",
                                      class_dictionary=None, debug=False):
    if os.path.exists(dir):
        annotations = open(os.path.join(dir, "annotations." + mode + ".csv"), "w")
        for root, _, files in os.walk(dir):
            if not root == dir:
                print(root + " -> class_id: " + str(class_dictionary[os.path.basename(root).replace("-open", "")]))
                t0 = time.clock()
            for file in files:
                if separator in file:
                    path = os.path.join(root, file.replace(separator, "frame").replace("png", "jpg"))
                    if os.path.isfile(path):
                        image_mask = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                        mask = cv2.inRange(image_mask, background_color1, background_color2)
                        if debug:
                            cv2.imshow("mask", mask)
                            cv2.waitKey(1)
                        indices = np.argwhere(mask[:, :] < 255)
                        if len(indices) > 0:
                            min = np.amin(indices, axis=0)
                            max = np.max(indices, axis=0)
                            min = (min[1], min[0])
                            max = (max[1], max[0])
                            label = os.path.basename(root)
                            if mode == "retinanet":
                                annotations.write("\"" + path + "\",")
                                annotations.write(
                                    str(min[0]) + "," + str(min[1]) + "," + str(max[0]) + "," + str(max[1]) + ",")
                                annotations.write(label.replace("-open", "") + "\n")
                            elif mode == "ssd":
                                annotations.write(label + "/" + file.replace(separator, "") + ",")
                                annotations.write(
                                    str(min[0]) + "," + str(min[1]) + "," + str(max[0]) + "," + str(max[1]) + ",")
                                annotations.write(str(class_dictionary[label.replace("-open", "")]) + "\n")
                        elif mode == "retinanet":
                            annotations.write("\"" + path + "\",,,,,\n")
            if not root == dir:
                t1 = time.clock()
                print("Time elapsed: ", t1 - t0)  # CPU seconds elapsed (floating point)
        annotations.close()
    else:
        raise ValueError("input dir not found!")


def parse_annotations_from_alpha_mask_multiclass(dir, separator, class_colors, nb_classes,
                                                 mode="retinanet", class_dictionary=None, debug=False):
    def clamp(n, smallest, largest):
        if n < smallest: return smallest
        if n > largest:
            return largest
        else:
            return n

    if os.path.exists(dir):
        annotations = open(os.path.join(dir, "annotations." + mode + ".csv"), "w")
        for file in os.listdir(dir):
            if file.endswith(".png"):
                if not os.path.isfile(os.path.join(dir, file)): continue
                if separator in file:
                    path = os.path.join(dir, file.replace(separator, "frame").replace("png", "jpg"))
                    if os.path.isfile(path):
                        image_mask = cv2.imread(os.path.join(dir, file), cv2.IMREAD_COLOR)
                        shape = image_mask.shape
                        w, h = shape[1], shape[0]
                        gt_found = False
                        for class_id, color in enumerate(class_colors):
                            m, n = 0.9, 1.1
                            mask = cv2.inRange(image_mask, m * np.array(color), n * np.array(color))
                            # blur image
                            mask = cv2.blur(mask, (2, 2))
                            # apply a threshold
                            mask = cv2.threshold(mask, 190, 250, cv2.THRESH_BINARY)
                            mask = mask[1]
                            indices = np.argwhere(mask[:, :] > 0)
                            min, max = None, None
                            if len(indices) > 200:
                                gt_found = True
                                min = np.amin(indices, axis=0)
                                max = np.max(indices, axis=0)
                                min = (clamp(min[1] - 1, 0, w), clamp(min[0] - 1, 0, h))
                                max = (clamp(max[1] + 1, 0, w), clamp(max[0] + 1, 0, h))
                                label = list(class_dictionary.keys())[class_id]
                                if mode == "retinanet":
                                    annotations.write("\"" + path + "\",")
                                    annotations.write(
                                        str(min[0]) + "," + str(min[1]) + "," + str(max[0]) + "," + str(max[1]) + ",")
                                    annotations.write(label.replace("-open", "") + "\n")
                                elif mode == "ssd":
                                    annotations.write(os.path.basename(path) + ",")
                                    annotations.write(
                                        str(min[0]) + "," + str(min[1]) + "," + str(max[0]) + "," + str(max[1]) + ",")
                                    annotations.write(str(1 + class_dictionary[label.replace("-open", "")]) + "\n")
                            if debug:
                                original = cv2.imread(path, cv2.IMREAD_COLOR)
                                if min:
                                    cv2.rectangle(original, min, max, (0, 255, 0))
                                cv2.imshow("original", original)
                                cv2.moveWindow("original", 0, 0)
                                cv2.imshow("mask", mask)
                                cv2.imshow("image_mask", image_mask)
                                cv2.moveWindow("mask", 500, 0)
                                cv2.moveWindow("image_mask", 1000, 0)

                                cv2.waitKey(0)
                        if (not gt_found) and mode == "retinanet":
                            annotations.write("\"" + path + "\",,,,,\n")
                        elif (not gt_found) and mode == "ssd":
                            annotations.write(os.path.basename(path) + ",0,0,0,0," + str(0) + "\n")
        annotations.close()
    else:
        raise ValueError("input dir not found!")


def patches(input_dir, output_dir):
    global pixels
    pixels = []
    window_name = "window"
    global original_dims
    original_dims = (0, 0)
    for dirpath, subdirs, files in os.walk(input_dir):
        for x in files:
            if x.endswith(".jpg") or x.endswith(".JPG") or x.endswith(".png") or x.endswith(".PNG"):
                class_name = os.path.basename(dirpath)
                img_path = os.path.join(dirpath, x)

                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, mark_pixel)
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
                    cv2.imshow(window_name, image)
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27 and len(pixels) > 0:
                        pixels.pop()

                    # if the 'c' key is pressed, break from the loop
                    elif key == ord("c"):
                        quit(0)
                    elif key == ord("n"):
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

                            width = abs(bb[0] - bb[2])
                            height = abs(bb[1] - bb[3])
                            diff = abs(width - height)
                            if width > height:
                                dims = width
                                bb[3] += diff
                                if float(bb[1]) - float(dims) / 2.0 >= 0:
                                    bb[1] -= float(dims) / 2.0
                                    bb[3] -= float(dims) / 2.0
                                else:
                                    bb[3] -= bb[1]
                                    bb[1] = 0
                                if bb[3] > original_dims[1]:
                                    bb[1] -= bb[3] - original_dims[1]
                                    bb[3] = original_dims[1]
                            else:
                                dims = height
                                bb[2] += diff
                                if float(bb[0]) - float(dims) / 2.0 >= 0:
                                    bb[0] -= float(dims) / 2.0
                                    bb[2] -= float(dims) / 2.0
                                else:
                                    bb[2] -= bb[0]
                                    bb[0] = 0
                                if bb[2] > original_dims[0]:
                                    bb[0] -= bb[2] - original_dims[0]
                                    bb[2] = original_dims[0]

                            crop_img = original_image[bb[1]:bb[3], bb[0]:bb[2]]
                            crop_img = cv2.resize(crop_img,
                                                  dsize=(int(crop_img.shape[1] * 0.25), int(crop_img.shape[0] * 0.25)))
                            cv2.imshow("cropped", crop_img)
                            cv2.waitKey(0)

                            pixels = []
                            break

                    image = clone.copy()


def shuffle_csv_annotations(file):
    input_file = open(file, "r")
    output_file = open(file.replace(".csv", ".shuffeled.csv"), "w")
    lines = []
    while True:
        line = input_file.readline()
        if line.strip() == "": break
        lines.append(line)
    import random
    random.shuffle(lines)
    for line in lines:
        output_file.write(line)


def repair_annotations(file):
    input = open(file, "r")
    output = open(file.replace(".csv",".reparied.csv"), "w")
    csv_reader = csv.reader(input, delimiter=',')
    result = {}
    for line, row in enumerate(csv_reader):
        path, x1, y1, x2, y2, class_id = row[:6]
        img_file = os.path.basename(path)
        if img_file not in result:
            result[img_file] = [(path, x1, y1, x2, y2, class_id)]
        else:
            result[img_file].append((result[img_file][0][0], x1, y1, x2, y2, class_id))
    input.close()
    for i, element in enumerate(result):
        for annot in result[element]:
            output.write("{path},{x1},{y1},{x2},{y2},{class_name}\n".format(path=annot[0],
                                                                            x1=annot[1],
                                                                            y1=annot[2],
                                                                            x2=annot[3],
                                                                            y2=annot[4],
                                                                            class_name=annot[5]))
    output.close()


if __name__ == '__main__':
    # input_dir = "D:/Documents/Villeroy & Boch - Subway 2.0"
    # annotations = "D:/Documents/Villeroy & Boch - Subway 2.0/annotations.csv"
    # labeling(input_dir, annotations)
    # debug_annotations("D:/Documents/3dsMax/renderoutput/3dsmax3/annotations.retinanet.csv")

    """
    classes = {
        "540000": 0,
        "5614R0": 1,
        "711355": 2,
        "7175A0": 3,
        "7175D0": 4,
        "751301": 5,
        "751301-open": 6,
        "5614R0-open": 7
    }
    colors = [
        (255,254,0),
        (185, 0, 255),
        (0,255,255),
        (0, 255, 185),
        (254, 0, 255),
        (0,0,255),
        (0,255,0),
        (255,186,0)
    ]
    parse_annotations_from_alpha_mask_multiclass(dir="D:/Documents/3dsMax/renderoutput/domain_randomization/test",
                                                 separator="objID",
                                                 mode="ssd",
                                                 class_dictionary=classes,
                                                 debug=False,
                                                 class_colors=colors,
                                                 nb_classes=6)
    """

    #shuffle_csv_annotations(file="D:/Documents/3dsMax/renderoutput/annotations.ssd.csv")
    #repair_annotations(file="D:/Documents/Villeroy & Boch - Subway 2.0/annotations.csv")
    debug_annotations("D:/Documents/3dsMax/renderoutput/annotations.ssd.shuffeled.csv", max_width=800, dir="D:/Documents/3dsMax/renderoutput")
