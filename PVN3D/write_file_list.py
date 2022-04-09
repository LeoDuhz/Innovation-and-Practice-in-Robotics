# file_path = "/home/zjunlict/dhz/NeuroRobotics/PVN3D/pvnet-rendering/data/LINEMOD/fuse/ape/file_list.txt"
file_path = "/home/zjunlict/dhz/Linemod_preprocessed/fuse/ape/file_list.txt"
f = open(file_path, "w")
f.truncate()
for i in range(5):
    num = "fuse/" + str(i) + "\n"
    # num = "/home/zjunlict/dhz/NeuroRobotics/PVN3D/pvnet-rendering/data/LINEMOD/renders/ape/" + str(i) + ".jpg" + "\n"
    f.write(num)

f.close()