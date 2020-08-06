import os, os.path
import wget
import cv2
import urllib.request
from time import sleep
from pathlib import Path

#  Set headers to not get 403: Forbidden
opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0"), ("Cache-Control", "no-store")]
urllib.request.install_opener(opener)
image_url = "https://thispersondoesnotexist.com/image"  # 1024x1024

#  Get n faces to download
def walkFiles():
    hi_files = next(os.walk(Path("./Data/High res")))[2]
    lo_files = next(os.walk(Path("./Data/Low res")))[2]
    return hi_files, lo_files


#  Get n faces in library
facecount = len([file for file in Path("./Data/High res").iterdir()])


#  Get n faces to download
command = input("Add or delete images? ['A' or 'D']\n")
if command.lower() == "a":
    try:
        extrafaces = int(input(f"Currently {facecount} faces available in './Data/High res'. How many to download?  [int]\n"))
    except Exception as e:
        print(e)
    i = facecount + 1

    #  Download requested n faces
    while i <= facecount + extrafaces:
        filename = "./Data/High res/face_{:0>5}.jpeg".format(i)
        image_filename = wget.download(image_url, filename)
        sleep(1)
        i += 1

    hi_files, lo_files = walkFiles()

    for img in hi_files:
        if img not in lo_files:
            cv2.imwrite(f"./Data/Low res/{img}", cv2.resize(cv2.resize(cv2.imread(f"./Data/High res/{img}", cv2.IMREAD_COLOR), (100, 100)), (1024, 1024)))
elif command.lower() == "d":
    delcmd = input("How many images do you want to purge? ['All' or int]\n")
    if delcmd.lower() == "all":

        hi_files, lo_files = walkFiles()

        for img in hi_files:
            os.remove(f"./Data/High res/{img}")
        for img in lo_files:
            os.remove(f"./Data/Low res/{img}")

    elif int(delcmd) <= facecount:
        hi_files, lo_files = walkFiles()

        i = 1
        while i <= int(delcmd):
            os.remove(f"./Data/High res/{hi_files[-i]}")
            os.remove(f"./Data/Low res/{lo_files[-i]}")
            i += 1
    else:
        print("Unvalid input")
else:
    print("Unvalid command")
