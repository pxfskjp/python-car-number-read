from tkinter import *

def processSettings(root_):
    root1 = Toplevel(root_)
    root1.geometry("600x300")
    root1.title(" Settings ")

    root = Frame(root1)
    root.pack(fill="both")
    
    def Take_input():
        AMQPurl = AMQP.get("1.0",'end-1c')
        rtspurl = rtsp.get("1.0",'end-1c')
        tokent = token.get("1.0",'end-1c')
        with open('AMQPurl.txt', 'w') as f:
            f.write(AMQPurl)
        with open('rtspurl.txt', 'w') as f:
            f.write(rtspurl)
        with open('token.txt', 'w') as f:
                f.write(tokent)


    l = Label(root,text="AMQP URL")
    AMQP = Text(root, height=2,
                    width=50,
                    bg="light yellow")
    lr = Label(root,text="rtsp URL")
    rtsp = Text(root, height=2,
                width=50,
                bg="light cyan")

    ltoken = Label(root,text="Company Token")
    token = Text(root, height=2,
                width=50,
                bg="light cyan")


    Display = Button(root, height=2,
                    width=20,
                    text="Save",
                    command=lambda: Take_input())
    Close = Button(root, height=2,
                    width=20,
                    text="Close"
                    , command=root.destroy)

    l1 = Label(root,text="")
    l2 = Label(root,text="")
    l3 = Label(root,text="")
    l4 = Label(root,text="")
    l5 = Label(root,text="")
    with open('AMQPurl.txt') as f:
        lines = f.readlines()
        AMQP.insert(1.0, lines)
    with open('rtspurl.txt') as f:
        lines = f.readlines()
        rtsp.insert(1.0, lines)
    with open('token.txt') as f:
        lines = f.readlines()
        token.insert(1.0, lines)

    l.grid(row=0, column=0, sticky="news")
    AMQP.grid(row=0, column=1, sticky="news")
    lr.grid(row=1, column=0, sticky="news")
    rtsp.grid(row=1, column=1, sticky="news")
    ltoken.grid(row=2, column=0, sticky="news")
    token.grid(row=2, column=1, sticky="news")
    l1.grid(row=3, column=0, sticky="news")
    l2.grid(row=4, column=0, sticky="news")
    l3.grid(row=5, column=0, sticky="news")
    l4.grid(row=6, column=0, sticky="news")
    l5.grid(row=7, column=0, sticky="news")



    Display.grid(row=8, column=0, sticky="news")
    #Close.grid(row=3, column=1, sticky="news")


    # l.pack()
    # AMQP.pack()
    # lr.pack()
    # rtsp.pack()
    ## Display.pack()
    # Close.pack()


    #mainloop()
