import tkinter as tk
from tkinter import ttk, filedialog, END, Listbox
from DiscreteAlphaModulation import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from functools import partial
from ltfatpy.sigproc.thresh import thresh
import scipy.io.wavfile as wave



######### processing ######

sig  = []
rate = 0
tt = []



def clearFrame(frame):
    # destroy all widgets from frame
    for widget in frame.winfo_children():
       widget.destroy()

def plot(window,y,z=None):

    clearFrame(window)
    # the figure that will contain the plot
    fig = Figure(figsize=(15, 6), dpi=100)

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(y)
    if z is not None:
        plot1.plot(z)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,    master=window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()


def applySig(window):
    global sig
    global rate
    sig = doppler(int(e1.get()), int(e2.get()), 2, int(e3.get()))
    rate = len(sig)
    e20.delete(0,END)
    e20.insert(0, len(sig))
    plot(window,sig)

def createNoise():

    global sig
    L     = len(sig)
    noise = np.random.normal(0, float(e4.get()), size=L)
    sig   = sig+noise
    plot(lowerframe,sig)

def applyGrid(window):
    global Ws,D

    Ws, D = frame_grid(int(e20.get()), eps=float(e21.get()), fs=rate, alp=float(e22.get()), plot=0, factor=int(e23.get()))

    clearFrame(window)
    fig = Figure(figsize=(15, 6), dpi=100)
    plot1 = fig.add_subplot(111)
    for u in range(len(D)):
        plot1.scatter(list(D.values())[u], [list(D.keys())[u]] * len(list(D.values())[u]), c='b', s=1)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()

def applyWind(window):

    global wind
    wind =   mySpline(int(e31.get()), int(e32.get()), int(e33.get()), 0)[1]
    plot(window,wind,sig)

def displaySpec(window):

    global tt
    tt = [mySTFT2(sig,wind,D[w],w,1/rate) for w in Ws]
    print(LIST.curselection())

    plt.figure()
    plotlist(tt, Ws,transList[LIST.curselection()[0]])
    plt.colorbar()

def denoise(window,th_method):

    global denoised_tt
    denoised_tt = [thresh(x.real, float(e51.get()), th_method)[0] + 1j * thresh(x.imag, float(e51.get()), th_method)[0] for x in tt]

    plt.figure()
    plotlist(denoised_tt, Ws)
    plt.colorbar()

def onLoad():

    global sig
    global rate

    ftypes = [('WAV files', '*.wav'), ('All files', '*')]
    dlg = filedialog.Open(filetypes=ftypes)
    fl = dlg.show()

    if fl != '':
        print(fl)
        (rate, sig) = wave.read(fl)
        sig = sig[:5*rate]
        plot(lowerframe,sig)
        e20.delete(0, END)
        e20.insert(0, len(sig))











root = tk.Tk()
root.wm_title("Embedding in Tk")
root.geometry("500x520")
root.title("Tab Widget")
tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)
tab5 = ttk.Frame(tabControl)

tabControl.add(tab1, text='Signal')
tabControl.add(tab3, text='Window')
tabControl.add(tab2, text='Grid')
tabControl.add(tab4, text='Spectrogramm')
tabControl.add(tab5, text='Denoising')
tabControl.pack(expand=1, fill="both")



# Frame1

frame = tk.Frame(tab1)
tk.Label(frame,text="Frequency").grid(row=0)
tk.Label(frame,text="Time").grid(row=1)
tk.Label(frame,text="Frequency sampling").grid(row=2)
tk.Label(frame,text="",width=50).grid(row=0,column=3)#,ipadx=150)
tk.Label(frame,text="noise sigma").grid(row=0,column=4)#,ipadx=150)

e1 = tk.Entry(frame)
e2 = tk.Entry(frame)
e3 = tk.Entry(frame)
e4 = tk.Entry(frame)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=0, column=5)

lowerframe = tk.Frame(tab1, bg='#80c1ff',bd=3)
lowerframe.place(relx=0.1,rely=0.4 ,relwidth=0.8,relheight=0.5)


tk.Button(frame,text='Apply',command=partial(applySig,lowerframe)).grid(row=3,column=0,sticky=tk.W,pady=4)
tk.Button(frame,text='Apply noise',command=createNoise).grid(row=1,column=4,sticky=tk.W,pady=4)
tk.Button(frame,text='Load audio',command=onLoad).grid(row=4,column=0,sticky=tk.W,pady=4)
frame.place(relx=0.1,rely=0.05 ,relwidth=0.8,relheight=0.3)




# Frame2

Ws ,D = [],[]

frame2 = tk.Frame(tab2)
tk.Label(frame2,text="length").grid(row=0)
tk.Label(frame2,text="eps").grid(row=1)
tk.Label(frame2,text="alp").grid(row=2)
tk.Label(frame2,text="factor").grid(row=3)

e20 = tk.Entry(frame2)
e21 = tk.Entry(frame2)
e22 = tk.Entry(frame2)
e23 = tk.Entry(frame2)

e20.grid(row=0, column=1)
e21.grid(row=1, column=1)
e22.grid(row=2, column=1)
e23.grid(row=3, column=1)

tk.Button(frame2,text='Apply',command= lambda :applyGrid(lowerframe2)).grid(row=4,column=0,sticky=tk.W,pady=4)
frame2.place(relx=0.1,rely=0.05 ,relwidth=0.8,relheight=0.3)

canvas2 = tk.Canvas(tab2,height=100,width=100)
lowerframe2 = tk.Frame(tab2, bg='#80c1ff',bd=3)
lowerframe2.place(relx=0.1,rely=0.4 ,relwidth=0.8,relheight=0.5)


# Frame3

wind = []


frame3 = tk.Frame(tab3)
tk.Label(frame3,text="B-splines").grid(row=0)
tk.Label(frame3,text="level").grid(row=1)
tk.Label(frame3,text="length").grid(row=2)
tk.Label(frame3,text="dillation").grid(row=3)

e31 = tk.Entry(frame3)
e32 = tk.Entry(frame3)
e33 = tk.Entry(frame3)

e31.insert(0, 4)
e32.insert(0, len(sig)//10)
e33.insert(0, 1)



e31.grid(row=1, column=1)
e32.grid(row=2, column=1)
e33.grid(row=3, column=1)


tk.Button(frame3,text='Apply',command=lambda :applyWind(lowerframe3)).grid(row=4,column=0,sticky=tk.W,pady=4)
frame3.place(relx=0.1,rely=0.05 ,relwidth=0.8,relheight=0.3)

canvas3 = tk.Canvas(tab3,height=100,width=100)
lowerframe3 = tk.Frame(tab3, bg='#80c1ff',bd=3)
lowerframe3.place(relx=0.1,rely=0.4 ,relwidth=0.8,relheight=0.5)

# Frame4

frame4 = tk.Frame(tab4)
tk.Label(frame4,text="Spectrogram").grid(row=0,column=1)

LIST = Listbox(frame4,selectmode = "unique")
LIST.grid(row=0,column=3)
transList = ['db','dbsq','linsq','linabs','lin']
[LIST.insert(END,x) for x in transList]



tk.Button(frame4,text='Display',command= lambda :displaySpec(lowerframe4)).grid(row=0,column=2,sticky=tk.W,pady=4)
frame4.place(relx=0.1,rely=0.05 ,relwidth=0.8,relheight=0.3)

canvas4 = tk.Canvas(tab4,height=100,width=100)
lowerframe4 = tk.Frame(tab4, bg='#80c1ff',bd=3)
lowerframe4.place(relx=0.1,rely=0.4 ,relwidth=0.8,relheight=0.5)



# Frame5

frame5 = tk.Frame(tab5)
tk.Label(frame5,text="universal threshold").grid(row=0)
tk.Label(frame5,text="threshold").grid(row=1)

e51 = tk.Entry(frame5)
e51.grid(row=1, column=1)




tk.Button(frame5,text='Apply hard thresholding',command= lambda :denoise(lowerframe5,'hard')).grid(row=4,column=0,sticky=tk.W,pady=4)
tk.Button(frame5,text='Apply soft thresholding',command= lambda :denoise(lowerframe5,'soft')).grid(row=5,column=0,sticky=tk.W,pady=4)

frame5.place(relx=0.1,rely=0.05 ,relwidth=0.8,relheight=0.3)

canvas4 = tk.Canvas(tab5,height=100,width=100)
lowerframe5 = tk.Frame(tab5, bg='#80c1ff',bd=3)
lowerframe5.place(relx=0.1,rely=0.4 ,relwidth=0.8,relheight=0.5)



root.mainloop()