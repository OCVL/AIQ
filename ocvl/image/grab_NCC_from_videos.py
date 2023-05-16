from tkinter import filedialog
import os, re, shutil


def get_files():
    f2 = filedialog.askopenfilenames()
    fileList = list(f2)
    return fileList


def get_directory():
    path = filedialog.askdirectory(title="Select the directory to start searching through.")
    return path


def find_files(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.endswith(".csv") and re.search(re.escape(filename) + r'[0-9]{1}.*', file):
                print(os.path.join(root, file))
                return os.path.join(root, file)


    # dir_list = os.listdir(search_path)
    # for i in dir_list:
    #     if i.endswith(".csv") and re.search(re.escape(filename) + r'*', i):
    #         return i


if __name__ == '__main__':
    direct = get_directory()
    f_list = get_files()
    save_dir = "P:/Brea_Brennan/Image_Quality_Analysis/ExtensionMaterials/Videos/NCC_files/"
    csv_files = []

    count = 0
    for f in f_list:
        print(f.split('/')[-1])
        new_name = f.split('/')[-1][:-4]
        print(new_name)
        csv_files.append(find_files(new_name, direct))
        shutil.copy(csv_files[count], save_dir)
        count += 1

    # shutil.copy(direct + "/" + csv_files[0], save_dir)
    print(csv_files)


