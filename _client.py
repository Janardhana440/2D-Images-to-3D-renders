# This is to be run on cmd or in any IDE you want idc
# search for BLEH to find locations where the path must be changed accordingly so that this works on your machine
# NOTE: for now, clear the contents of client_shape.json and shape.json before every run.
# You can replace them directly with the contents of the template files
# to be automated

from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import cv2
import socket
import json
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk     # from tkinter import Tk for Python 3.x
import pytesseract
import temp
ocr_vals = []
Y = 0


def get_image_details(path):

    img = cv2.imread(path)  # BLEH! CHANGE PATH ACCORDINGLY!!!
    height = img.shape[0]
    width = img.shape[1]
    data_to_json = {
        "img_path": path,
        "img_height": height,
        "img_width": width
    }
    write_json(data_to_json, obj_name="image_details")


def clear_json(path, template):
    with open(template, "r") as t, open(path, "w") as p:
        p.write(t.read())


def list_to_str(l):
    s = ""
    for i in l:
        s = s + str(i) + "$"
    return s


def select_image():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename()
    return filename


def start_up(img_path):
    print("cleaning up output-images folder...")
    current_working_directory = os.getcwd()
    # DO NOT CHANGE output_images_path UNDER ANY CIRCUMSTANCES!!!

    # output_images_path = r"C:\Users\rbhan\Downloads\Capstone\output-images"  # BLEH
    # output_images_path = r"C:\Users\jackb\OneDrive\Documents\server_new\saved_imgs"
    output_images_path = 'saved_imgs'
    if output_images_path != current_working_directory:
        os.chdir(output_images_path)
        all_files = os.listdir()
        for f in all_files:
            os.remove(f)
        print("cleaning up complete.")
    else:
        print("why do you want to erase all of our hard work")

    print("the image that is being used is ", img_path)
    os.chdir(current_working_directory)
    conf = input("Would you like to continue? y/n\t:\t")
    if conf == 'n' or "":

        exit()


def imag_pro(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    plt.imshow(img)
    plt.show()
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    print("image path is ", img_path)
    return img_preprocessed


def detect_shapes(img_path):
    directory = os.walk(img_path)
    files = []
    for root, dir, fil in directory:
        for file in fil:
            print(file)
            files.append(os.path.join(root, file))
    # model1 = keras.models.load_model(r'C:\Users\rbhan\Downloads\Capstone\model\CNN_aug_best_weights.h5')  # BLEH! CHANGE PATH ACCORDINGLY!!!
    # BLEH! CHANGE PATH ACCORDINGLY!!!
    # model1 = keras.models.load_model(
    #     r"C:\Users\jackb\OneDrive\Documents\capstone img classifier\CNN_aug_best_weights.h5")

    model1 = keras.models.load_model(
        r'F:\Actual Projects\capstone_server_final\model\CNN_aug_best_weights.h5')
    classNames = ['circle', 'pentagon', 'square', 'triangle']
    test_preprocessed_images = np.vstack([imag_pro(fn) for fn in files])
    prediction = model1.predict(test_preprocessed_images)

    arr = np.argmax(prediction, axis=1)
    # print("arr is ", arr)
    final = []

    for i in arr:
        final.append(classNames[i])
    return final


def write_json(new_data, filename='json_data\shape.json', obj_name="object_details"):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[obj_name].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


def extract_info(final_list, Y):
    ThreeDEquivalent = {'circle': "sphere", 'pentagon': "pentagonal_prism",
                        'square': "cube", 'triangle': "triangular_prism"}
    print("EXTRACT INFO")
    msg = []
    print(final_list, "\n")
    for tup in final_list:
        print(tup)
        shape_name = tup[0]
        l = tup[1].split('_')
        print(l)
        x_coord = str(int(l[3]) / 1)  # 5)
        y_coord = str(int(l[5]) / 1)  # 5)
        print("XY CORDS")
        width = str(int(l[7]))
        height = str(int(l[9][:-4]))

        msg.append(ThreeDEquivalent[shape_name] +
                   " " + x_coord + " " + y_coord + " 0 " + width + " " + height)
    return msg


def send_ocr_vals(ocr_list):
    # this function sends all ocr values to blender
    # later on in blender, looking at the positions of these ocr values and the shapes being generated
    #   1) values such as weight can be added in looking at proximity
    #   2) any text (like m1) can be thrown in a text box or loaded onto the shape as a texture (probably not gonna do the texture part)

    host = socket.gethostname()  # as both code is running on same pc
    port = 5001  # socket server port number for OCR messaging

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    for msg in ocr_list:
        print("sending ocr values :  ", msg)
        message = list_to_str(msg)
    # okay here's hoping
        print(message)
        while message.lower().strip() != '':
            client_socket.send(message.encode())  # send message
            message = input(" -> ")  # again take input

    client_socket.close()  # close the connection


def client_program(img_path):
    global ocr_vals, Y
    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    img = cv2.imread(img_path)  # BLEH! CHANGE PATH ACCORDINGLY!!!
    Y, width12 = img.shape[:2]

    # image = cv2.imread("F:\Actual Projects\Capstone\images\p1.jpg")
    image = img

    new_image = image.copy()

    # lfggggg. what are the numbers now?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display grayscale
    ret, binary = cv2.threshold(gray, 100, 255,
                                cv2.THRESH_OTSU)

    # invert colours
    inverted_binary = ~binary

    # Find contours and store in list

    contours, hierarchy = cv2.findContours(inverted_binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Contours to red color
    with_contours = cv2.drawContours(image, contours, -1, (255, 0, 255), 3)

    # total no of contours detected
    print("here2")
    print('Total number of contours detected: ' + str(len(contours)))

    # bounding box around the third contour
    x, y, w, h = cv2.boundingRect(contours[len(contours)-1])
    cv2.rectangle(with_contours, (x, y), (x+w, y+h), (255, 0, 0), 5)
    cv2.imshow('Third contour with bounding box', with_contours)
    x_coord = []
    y_coord = []
    width = []
    height = []
    a = 0
    ROI_number = 0
    prevx = 0
    # Draw a bounding box around all contours and crop it out into seperate image files
    print("here4")
    for c in contours:
        if (cv2.contourArea(c)) > 1000:
            cv2.rectangle(with_contours, (x, y), (x+w, y+h), (255, 0, 0), 5)
            x, y, w, h = cv2.boundingRect(c)
            print(x, y, w, h)
            x_coord.append(x)  # from bottom right -> top left corner
            y_coord.append(y)
            width.append(w)
            height.append(h)
            a = x_coord[-1]
            if abs(a-prevx) <= 9:
                x_coord.pop(-1)
                y_coord.pop(-1)
                height.pop(-1)
                width.pop(-1)
            prevx = a

    print("x coords are  ", x_coord)
    print("y coord is ", y_coord)
    print("height is ", height)
    print("width is ", width)
    new_img = cv2.imread(img_path)
    # okay so now its a matter of sending these numbers over to blender, using the sever.py file?
    # so maybe i can copy this code intoo that file, and then change the vars?
    # okay lets see
    # image = cv2.imread("F:\Actual Projects\Capstone\images\p1.jpg")
    for i in range(0, len(x_coord)):
        print(y_coord[i], height[i], x_coord[i], width[i])
        ROI = new_img[y_coord[i]:y_coord[i]+height[i],
                      x_coord[i]:x_coord[i]+width[i]]
        # cv2.imwrite(
        #     r'C:\Users\rbhan\Downloads\Capstone\output-images\ROI_{}_x_{}_y_{}_dim_{}.png'.format(
        #         ROI_number, x_coord[i], y_coord[i], (height[i]+width[i])//2), ROI)
        cv2.imwrite('saved_imgs\ROI_{}_x_{}_y_{}_w_{}_h_{}.png'.format(
            ROI_number, x_coord[i], Y-y_coord[i], width[i], height[i]), ROI)
        # cv2.rectangle(new_image,(x,y),(x+w,y+h),(36,255,12),2)
        ROI_number += 1

    cv2.imshow('All contours with bounding box', with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # same thing for xcoord and ycoord?

    # OCR MAN OCR MAN OCR MAN

    shape_list = detect_shapes(r"saved_imgs")
    # file_list = os.listdir(r"C:\Users\rbhan\Downloads\Capstone\output-images")
    file_list = os.listdir(r"saved_imgs")
    # ideally this is the only thing we will need to finally build our shapes
    ocr_vals = temp.ocrf(img_path, Y)

    final_list = list(zip(shape_list, file_list))
    print("HELLLOOOOOOO FINAL LIST HERE")
    print(final_list)
    fin = extract_info(final_list, Y)
    # sending shape details here
    for msg in fin:
        print("building ", msg)
        message = msg
    # okay here's hoping
        print(message)
        while message.lower().strip() != '':
            print("writing to the json file (before sending it to client program)")

            shape,  x, y, z, w, h = message.split(' ')
            data_to_json = {
                "shape": shape,
                "x": x,
                "y": y,
                "z": z,
                "w": w,
                "h": h
            }
            write_json(
                data_to_json, filename='json_data\shape.json', obj_name="object_details")
            print("finished writing data to json file (before sending to client program)")

            client_socket.send(message.encode())  # send message
            data = client_socket.recv(1024).decode()  # receive response

            print('Received from server: ' + data)  # show in terminal

            message = input(" -> ")  # again take input

    client_socket.close()  # close the connection

    # sending ocr details
    send_ocr_vals(ocr_vals)


# print("input is in format --------> 'shape' 'dim' 'x' 'y' 'z'")
if __name__ == '__main__':
    global img_path

    # ocr_template_path = r"F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\templates\ocr_template.json"
    # shape_template_path = r"F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\templates\shape_template.json"
    ocr_template_path = 'templates\ocr_template.json'
    shape_template_path = 'templates\shape_template.json'

    ocr_json_path = 'json_data\ocr.json'
    shape_json_path = 'json_data\shape.json'
    clear_json(ocr_json_path, ocr_template_path)
    clear_json(shape_json_path, shape_template_path)
    print("cleared")
    img_path = select_image()  # BLEH
    get_image_details(img_path)
    start_up(img_path)
    client_program(img_path)
