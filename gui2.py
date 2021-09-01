from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
from threading import Thread
import time
import pickle
import csv
from datetime import datetime
import pika
from detect import initMain
from gui_mq import receiveMessage
from Settings import processSettings

mq_thread = ""

def sendmessage(carplatenumber):
    url2 =""
    with open('AMQPurl.txt') as f:
        url2 = f.readlines()
    print(url2)
    url = os.environ.get('CLOUDAMQP_URL', url2[0])
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()  # start a channel
    channel.queue_declare(queue='CarCam')  # Declare a queue
    channel.basic_publish(exchange='',
                          routing_key='CarCam',
                          body=carplatenumber)

    print(" [x] Sent '"+ carplatenumber+ "'")
    connection.close()

root = Tk()
root.geometry("1000x630")
root.resizable(False, False)
root.title('Netherlands,Belgium ALPR APP')

# parent frame of all
mainFrame = Frame(root)
mainFrame.pack(fill="both")

# string variable for label of bottom widget
bottomWidgetLabel_var = StringVar()
bottomWidgetLabel_var.set("Infos will be shown here")

# detected frame stored as this image
recognizedImagePath = os.path.abspath("./tempImg/temp.jpg")

def refresh():
    global root
    root.after(300, root.update)

# write csv
def writeToCsv(path):
    headers = ['input path', 'detections', 'time']
    filename = 'detections.csv'

    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")

        # write the data
        detected_number_plates = dynamicListBox.get(0, END)
        # detected_number_plates = pickle.load( open( "detected_number_plates.p", "rb" ) )
        detected_number_plates_str = ':'.join(detected_number_plates)
        data = {'input path': path, 'detections': detected_number_plates_str, 'time': str(current_time)}
        print(data)
        writer.writerow(data)


# recognizer function
def callRecognizer(path):


    pickle.dump(set(), open("detected_number_plates.p", 'wb'))
    pickle.dump({"done": False}, open("done.p", 'wb'))
    pickle.dump({"stop": False}, open("stop.p", 'wb'))
    dynamicListBox.delete(0, END)
    #command = "python detect.py --weights weights/best_new.pt --img 416 --conf 0.4 --source {}".format(path)
    params = "--weights weights/best_new.pt --img 416 --conf 0.4 --source {}".format(path)
    print( params )
    params= {'weights': 'weights/best_new.pt', 'imgsz':416, 'source':path}
    initMain(opt_=params)
    # command = "detect --weights weights/best_new.pt --img 416 --conf 0.4 --source {}".format(path)
    #os.system(command)
    while True:
        if os.path.getsize("stop.p") > 0:
            stop = pickle.load(open("stop.p", "rb"))
            if stop["stop"]:
                break
    # bottomWidgetLabel_var.set("Image recognition done")


# show image from path to canvas
def showImageInCanvas(path):
    global dynamicImage
    basewidth = 700
    img = Image.open(path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    dynamicImage = ImageTk.PhotoImage(img)
    # dynamicImageLabel = Label(root)
    dynamicImageCanvas.create_image(0, 0, anchor=NW, image=dynamicImage)
    root.after(5000, root.update)


def threadedImageRendering(pickedFileName):
    pickle.dump({"stop": False}, open("stop.p", 'wb'))

    if os.path.exists(recognizedImagePath):
        os.remove(recognizedImagePath)
    showImageInCanvas(pickedFileName)
    bottomWidgetLabel_var.set("Starting image recognition")

    while True:
        if os.path.getsize("stop.p") > 0:
            stop = pickle.load(open("stop.p", "rb"))
            if stop["stop"]:
                break
        if os.path.exists(recognizedImagePath):
            time.sleep(1)
            img = cv2.imread(recognizedImagePath)
            print(img.shape)
            showImageInCanvas(recognizedImagePath)
            bottomWidgetLabel_var.set("Loaded Recognized Image")
            detected_number_plates = pickle.load(open("detected_number_plates.p", "rb"))
            print(124)
            sendmessage(str(detected_number_plates))

            for i in detected_number_plates:
                dynamicListBox.insert(END, i)
            work = pickle.load(open("done.p", "rb"))
            if work["done"]:
                bottomWidgetLabel_var.set("Image Recognition Done")
                writeToCsv(pickedFileName)
                break


def threadedVideoRendering(pickedFileName):
    pickle.dump({"stop": False}, open("stop.p", 'wb'))

    if os.path.exists(recognizedImagePath):
        os.remove(recognizedImagePath)
    bottomWidgetLabel_var.set("Starting video stream recognition")

    # cap = cv2.VideoCapture(pickedFileName)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("Total frames: {}".format(total_frames))
    frame_count = 0

    while True:
        # refresh()
        if os.path.getsize("stop.p") > 0:
            stop = pickle.load(open("stop.p", "rb"))
            if stop["stop"]:
                break
        if os.path.exists(recognizedImagePath):
            frame_count += 1
            try:
                showImageInCanvas(recognizedImagePath)
                time.sleep(0.8)
            except:
                print("frame skipped")
            bottomWidgetLabel_var.set("Loaded Recognized Video Frames")
            detected_number_plates = pickle.load(open("detected_number_plates.p", "rb"))
            print(163)
            if len(detected_number_plates)>0:
                sendmessage(str(detected_number_plates))
            currentlyShowing = dynamicListBox.get(0, END)
            for i in detected_number_plates:
                if i not in currentlyShowing:
                    dynamicListBox.insert(END, i)
            work = pickle.load(open("done.p", "rb"))
            if frame_count % 500 == 0:
                root.update()
            if work["done"]:
                bottomWidgetLabel_var.set("Video Recognition Done")

                writeToCsv(pickedFileName)
                break


# top widget functions
def imagePickerButtonPressed():
    pickedFileName = filedialog.askopenfilename(title="Pick an image", filetypes=(
    ("Png files", "*.png"), ("Jpeg files", "*.jpeg"), ("Jpg files", "*.jpg")))
    print("")
    print(recognizedImagePath)
    if pickedFileName:
        Thread(target=threadedImageRendering, args=(pickedFileName,)).start()
        Thread(target=callRecognizer, args=(pickedFileName,)).start()


def videoPickerButtonPressed():
    pickedFileName = filedialog.askopenfilename(title="Pick an Video", filetypes=(("Mp4 files", "*.mp4"),))
    if pickedFileName:
        Thread(target=threadedVideoRendering, args=(pickedFileName,)).start()
        Thread(target=callRecognizer, args=(pickedFileName,)).start()

def getMessageFromMQ():
    print("**"*20)
    print("Calling the receivemessage function")
    global mq_thread
    mq_thread = Thread(target=receiveMessage, args=(getDataFromMQ,))
    mq_thread.start()
    #receiveMessage(getDataFromMQ)

def getDataFromMQ(ch, method, properties, body):
    global mq_thread
    from tkinter import messagebox
    messagebox.showinfo("Message from MQ", str(body))
    print("=="*20)
    ch.close()
    mq_thread.join()

def startRecognitionButtonPressed():
    #rtspurl = rtspInput.get()
    rtspurl=""
    with open('rtspurl.txt') as f:
        rtspurl = f.readlines()
    rtspurl=rtspurl[0]
    if "rtsp://" in rtspurl:
        vcap = cv2.VideoCapture(rtspurl)
        if vcap.isOpened():
            Thread(target=threadedVideoRendering, args=(rtspurl,)).start()
            Thread(target=callRecognizer, args=(rtspurl,)).start()
        else:
            bottomWidgetLabel_var.set("rtsp url is not open")
    else:
        bottomWidgetLabel_var.set("rtsp url is not correct")


# top widget functions end

# bottom widget functions

def stopAllPressed():
    pickle.dump({"stop": True}, open("stop.p", 'wb'))


# bottom widget functions end


# top widgets
topWidget = Frame(mainFrame)

# image file picker
imagePickerButton = Button(topWidget, text="Pick Image", command=imagePickerButtonPressed, fg="blue")
imagePickerButton.grid(row=0, column=0, sticky="news")

# video file picker
videoPickerButton = Button(topWidget, text="Pick Video", command=videoPickerButtonPressed, fg="blue")
videoPickerButton.grid(row=0, column=1, sticky="news")

# rtsp input
rtspInput = Entry(topWidget, width=70)
rtspInput.insert(0, "Enter rtsp url")
rtspInput.configure(state=DISABLED)

def Take_input():
    processSettings(root)
    #command = "python Settings.py"
    #os.system(command)

Display = Button(root, height=2,
                 width=20,
                 text="Settings",
                 command=lambda: Take_input())
Settings = Button(topWidget, text="Settings", command=Take_input, fg="blue")
Settings.grid(row=0, column=3, sticky="news")


def rtspInputOnClick(event):
    rtspInput.configure(state=NORMAL)
    rtspInput.delete(0, END)
    rtspInput.unbind('<Button-1>', on_click_id)


on_click_id = rtspInput.bind('<Button-1>', rtspInputOnClick)

# start recognition button
startRecognitionButton = Button(topWidget, text="Start Recognition", command=startRecognitionButtonPressed, fg="blue")
startRecognitionButton.grid(row=0, column=2, sticky="news")

mqButton = Button(topWidget, text="MQ Messages", command=getMessageFromMQ, fg="blue")
mqButton.grid(row=0, column=4, sticky="news")

# top widget column width config

topWidget.grid_columnconfigure(0, weight=1)
topWidget.grid_columnconfigure(1, weight=1)
topWidget.grid_columnconfigure(2, weight=3)
topWidget.grid_columnconfigure(3, weight=1)
topWidget.pack(fill="x", padx=10, pady=10)
# top widget ends


# middle widget
middleWidget = Frame(mainFrame)

# dynamic image widget
canvasHolder = Frame(middleWidget, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
canvasHolder.grid(row=0, column=0, sticky="news")

dynamicImageCanvas = Canvas(canvasHolder)
dynamicImageCanvas.pack(fill="both", expand=True)

# middleWidgetRightabel = Label(middleWidget,text="Detected Number Plates",fg="blue")
# middleWidgetRightabel.grid(row=0,column=1)

dynamicListBox = Listbox(middleWidget)
dynamicListBox.grid(row=0, column=1, sticky="nwes")

# middle widget column width config

middleWidget.grid_columnconfigure(0, weight=2)
middleWidget.grid_columnconfigure(1, weight=1)
middleWidget.grid_rowconfigure(0, minsize=500)

middleWidget.pack(fill="both", expand=True, padx=10, pady=10)

# middle frame widget ends


# bottom frame widget

bottomWidget = Frame(mainFrame)
dynamicBottomLabel = Label(bottomWidget, textvariable=bottomWidgetLabel_var, fg="blue")
dynamicBottomLabel.grid(row=0, column=0, sticky="w")

# stop button
bottomStopButton = Button(bottomWidget, text="Stop Recognition", command=stopAllPressed, fg="blue")
bottomStopButton.grid(row=0, column=1, sticky="e")

bottomWidget.grid_columnconfigure(0, weight=4)
bottomWidget.grid_columnconfigure(1, weight=1)
bottomWidget.pack(fill="x", padx=10, pady=10)




#Display.pack()

root.mainloop()
