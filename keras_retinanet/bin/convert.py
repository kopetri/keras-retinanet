import cv2

if __name__ == '__main__':
    annotation_file = "D:/Documents/Villeroy & Boch - Subway 2.0/annotations.csv"
    content = []
    with open(annotation_file, "r") as file:
        for line in file:
            content.append(line)
    with open(annotation_file, "w") as file:
        for line in content:
            data = line.split(",")
            if not data[1] == "":
                shape = cv2.imread(data[0]).shape
                if int(data[1]) < 7:
                    data[1] = str(0)
                elif int(data[1]) > shape[1] - 7:
                    data[1] = str(shape[1])
                if int(data[3]) < 7:
                    data[3] = str(0)
                elif int(data[3]) > shape[1] - 7:
                    data[3] = str(shape[1])
                if int(data[2]) < 7:
                    data[2] = str(0)
                elif int(data[2]) > shape[0] - 7:
                    data[2] = str(shape[0])
                if int(data[4]) < 7:
                    data[4] = str(0)
                elif int(data[4]) > shape[0] - 7:
                    data[4] = str(shape[0])
                line = ",".join(data)
            file.write(line)
