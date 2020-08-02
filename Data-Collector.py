import os, os.path
import wget
import cv2
import urllib.request
from time import sleep

#  Set headers to not get 403: Forbidden
opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0"), ("Cache-Control", "no-store")]
urllib.request.install_opener(opener)
image_url = "https://thispersondoesnotexist.com/image"  # 1024x1024

#  Get n faces to download
path, dirs, files = next(os.walk("./Data/High res"))
facecount = len(files)

command = input("'add' or 'del' ?  ")
if command == "add":
    try:
        extrafaces = int(input(f"Currently {facecount} faces available in './Data/High res'. How many more faces to download?  "))
    except Exception as e:
        print(e)
    i = facecount + 1

    #  Download requested n faces
    while i <= facecount + extrafaces:
        filename = "./Data/High res/face_{:0>5}.jpeg".format(i)
        image_filename = wget.download(image_url, filename)
        sleep(1)
        i += 1


    t, t, hi_files = next(os.walk("./Data/High res"))
    t, t, lo_files = next(os.walk("./Data/Low res"))

    for img in hi_files:
        if img not in lo_files:
            cv2.imwrite(f"./Data/Low res/{img}", cv2.resize(cv2.resize(cv2.imread(f"./Data/High res/{img}", cv2.IMREAD_COLOR), (100, 100)), (1024, 1024)))
elif command == "del":
    pass