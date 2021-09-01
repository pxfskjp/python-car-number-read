
from tkinter import filedialog


from threading import Thread

import csv
from datetime import datetime


import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from PaddleOCR.paddleocr import PaddleOCR

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import os
from tkinter import *
from PIL import Image, ImageTk
# ocr


ocr = None  # need to run only once to load model into memory

import pickle

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import increment_path, set_logging, check_requirements, check_img_size, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, save_one_box, colorstr, strip_optimizer
from utils.plots import plot_one_box, colors
from utils.torch_utils import select_device, load_classifier, time_sync

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
ocr = PaddleOCR(lang='en')
@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam

        ):

    imgsz = 640  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = 'runs/detect'  # save results to project/name
    name = 'exp'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
        print("dataset size: {}".format(bs))
    else:
        # view_img = check_imshow()
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if os.path.getsize("stop.p") > 0:
            stop = pickle.load(open("stop.p", "rb"))
            if stop["stop"]:
                break

        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        print("predictions  size: {}".format(len(pred)))
        print(pred)
        # Process predictions
        for i, det in enumerate(pred):
            print(det)  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count

            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            print("detections size: {}".format(len(det)))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        im0 = plot_one_box(ocr, xyxy, im0, label=label, color=colors(c, True),
                                           line_width=line_thickness)

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                frame = cv2.resize(im0, (600, 400))

                cv2.imshow("Number Plate Recognizing", frame)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite("./tempImg/temp.jpg", im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    cv2.imwrite("./tempImg/temp.jpg", im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
    pickle.dump({"done": True}, open("done.p", 'wb'))



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
    command = "python detect.py --weights weights/best_new.pt --img 416 --conf 0.4 --source {}".format(path)
    run(weights ='weights/best_new.pt' ,source=path)
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


def startRecognitionButtonPressed():
    rtspurl = rtspInput.get()
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
rtspInput.grid(row=0, column=2, sticky="news")


def rtspInputOnClick(event):
    rtspInput.configure(state=NORMAL)
    rtspInput.delete(0, END)
    rtspInput.unbind('<Button-1>', on_click_id)


on_click_id = rtspInput.bind('<Button-1>', rtspInputOnClick)

# start recognition button
startRecognitionButton = Button(topWidget, text="Start Recognition", command=startRecognitionButtonPressed, fg="blue")
startRecognitionButton.grid(row=0, column=3, sticky="news")

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

root.mainloop()