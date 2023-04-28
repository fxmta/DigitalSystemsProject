
import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image

file_types = [("JPEG (*.jpg)", "*.png"),
              ("All files (*.*)", "*.*")]
factor = 1

def tomato():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 88, 85], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 == 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)

    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 7], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2 / number_of_white_pix_1) * 100 * 0.2)
    global percentage
    percentage = 100 - percentage1

    if (percentage < 0):
        percentage = 0
    print('percentage of ripness = ', percentage)

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)

def strawberry():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 88, 85], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 == 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)

    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 6], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2 / number_of_white_pix_1) * 100 * 0.2)
    global percentage
    percentage = 100 - percentage1

    if (percentage < 0):
        percentage = 0
    print('percentage of ripness = ', percentage)

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)


def orange():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 81, 97], dtype="uint8")
    upper = np.array([40, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 = 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)
    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 6], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2 / number_of_white_pix_1) * 100 * factor)
    global percentage
    percentage = 100 - percentage1

    if (percentage < 0):
        percentage = 0
    print('percentage of ripness = ', percentage)

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)

def mango():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([12, 93, 0], dtype="uint8")
    upper = np.array([35, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 == 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)

    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 6], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2 / number_of_white_pix_1) * 100 * factor)
    global percentage
    percentage = 100 - percentage1
    print('percentage of ripness = ', percentage)

    if (percentage < 0):
        percentage = 0

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)


def apple():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 20], dtype="uint8")
    upper = np.array([10, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 == 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)

    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 6], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2 / number_of_white_pix_1) * 100 * factor)
    global percentage
    percentage = 100 - percentage1

    if(percentage<0):
        percentage=0
    print('percentage of ripness = ', percentage)

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)

def banana():
    image = cv2.imread('frame1.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([12, 93, 0], dtype="uint8")
    upper = np.array([35, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_1 = np.sum(mask == 255)
    if (number_of_white_pix_1 == 0):
        number_of_white_pix_1 == 0.001
    print('Number of white pixels:', number_of_white_pix_1)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original)

    image2 = cv2.imread('frame1.png')
    original2 = image2.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    lower = np.array([32, 86, 6], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask2 = cv2.inRange(image2, lower, upper)
    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original2, (x, y), (x + w, y + h), (36, 255, 12), 2)
    number_of_white_pix_2 = np.sum(mask2 == 255)
    if (number_of_white_pix_2 == 0):
        number_of_white_pix_2 == 0.001
    print('Number of white pixels:', number_of_white_pix_2)
    percentage1 = round((number_of_white_pix_2/number_of_white_pix_1)*100*factor)
    global percentage
    percentage = 100-percentage1
    print('percentage of ripness = ', percentage)

    if (percentage < 0):
        percentage = 0

    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('original2.png', original2)

    #cv2.imshow('mask', mask)
    #cv2.imshow('original', original)

def main():

    #sg.theme('BlueMono')
    sg.theme('BluePurple')

    camera_viewer_column = [[sg.Text('CAMERA VIEW', size=(40, 1), justification='center', font='Helvetica 14')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Test', size=(10, 1), font='Helvetica 14'),
               sg.Button('Cancel', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14'),
               sg.Button('Reset', size=(10, 1), font='Helvetica 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'),
               ],
                [sg.Text("  ")],
                [sg.Text("  "), sg.Radio('APPLE', "RADIO1", default=True, key="APPLE"),
                sg.Text("  "), sg.Radio('BANANA', "RADIO1", default=False, key="BANANA"),
                sg.Text("  "), sg.Radio('MANGO', "RADIO1", default=False, key="MANGO"),
                sg.Text("  "), sg.Radio('STRAWBERRY', "RADIO1", default=False, key="STRAWBERRY"),
                sg.Text("  "), sg.Radio('ORANGE', "RADIO1", default=False, key="ORANGE"),
                sg.Text("  "), sg.Radio('TOMATO', "RADIO1", default=False, key="TOMATO")],
                [sg.Text("  ")],
                [sg.Text('COUNT =', size=(8, 1), font=('Helvetica', 20),text_color='black'),
                 sg.Text('00', size=(10, 1), font=('Helvetica', 20), text_color='black', key='input')],

                [sg.Text('PERCENT =', size=(10, 1), font=('Helvetica', 20), text_color='black'),
                 sg.Text('00', size=(10, 1), font=('Helvetica', 20), text_color='black', key='input1')]
                            ]

    image_viewer_column = [
        #[sg.Text("IMAGE TAKEN FROM CAMERA:", size=(20, 1), justification='center', font='Helvetica 14')],
        #[sg.Text(size=(20, 1), key="-TEXT-")],
        [sg.Image("myimage.png", key="IMAGE1"),
         sg.Image("myimage.png", key="IMAGE2")],
        [sg.Image("myimage.png", key="IMAGE3"),
         sg.Image("myimage.png", key="IMAGE4")],
        #[sg.Text('00', size=(15, 1), font=('Helvetica', 40),text_color='black', key='input')],
    ]

    layout = [
        [sg.Column(camera_viewer_column),
         sg.VSeperator(),
         sg.Column(image_viewer_column),
         ]
    ]

    # create the window and show it without the plot
    window = sg.Window('FRUIT RIPNESS DETECTION',layout, location=(0,0), size=(1400,600), resizable=True).Finalize()
    window.Maximize()

    cap = cv2.VideoCapture(0)
    recording = True
    count = 0

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Test':
            recording = True
            img_name = "frame1.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            count = count+1
            print(count)
            window.FindElement('input').Update(count)

            if values["APPLE"] == True:
                print("apple")
                apple()
                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

            elif values["BANANA"] == True:
                print("banana")
                banana()
                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

            if values["MANGO"] == True:
                print("mango")
                mango()
                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

            elif values["STRAWBERRY"] == True:
                print("strawberry")
                strawberry()
                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

            elif values["ORANGE"] == True:
                print("orange")
                orange()

                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

            elif values["TOMATO"] == True:
                print("tomato")
                tomato()
                image = Image.open('original.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE1"].update(filename=filename)

                image = Image.open('mask.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE2"].update(filename=filename)

                image = Image.open('original2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE3"].update(filename=filename)

                image = Image.open('mask2.png')
                new_image = image.resize((400, 300))
                new_image.save('myimage.png')
                filename = "myimage.png"
                window["IMAGE4"].update(filename=filename)

                window.FindElement('input1').Update(percentage)

        elif event == 'Stop':
            recording = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)

        elif event == 'Cancel' and count >= 1:
            count = count-1
            window.FindElement('input').Update(count)

        elif event == 'Reset':
            count = 0
            window.FindElement('input').Update(count)

        if recording:
            ret, frame = cap.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

main()