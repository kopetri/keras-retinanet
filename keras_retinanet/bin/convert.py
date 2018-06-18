if __name__ == '__main__':
    annotation_file = "D:/Documents/dataset/villeroy-boch-black-noreflection/annotations.csv"
    content = []
    with open(annotation_file, "r") as file:
        for line in file:
            content.append(line)
    with open(annotation_file, "w") as file:
        for line in content:
            print(line)
            data = line.split(",")
            if not data[1] == "":
                data[1] = str(int(min(512, max(0, round(float(data[1]))))))
                data[2] = str(int(min(512, max(0, round(float(data[2]))))))
                data[3] = str(int(min(512, max(0, round(float(data[3]))))))
                data[4] = str(int(min(512, max(0, round(float(data[4]))))))
                line = ",".join(data)
            file.write(line)
