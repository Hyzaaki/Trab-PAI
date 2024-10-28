from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import scipy.io
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import svm
from sklearn.model_selection import cross_val_score 

# Calcular NT
NT = (661405 + 779052) % 4  

if NT == 0:
    # Calcular características de textura coarseness, contrast, periodicity e roughness
    pass  
elif NT == 1:
    # Calcular características LBP energia e entropia
    pass  
elif NT == 2:
    # Calcular momentos invariantes de Hu
    pass  
elif NT == 3:
    # Calcular descritores de Tamura
    pass  
#______________________________________________

# Variáveis globais
is_img_loaded = False
file_path = ""
loaded_img = None
no_loaded_text_exists = True
roi_menu = None
#______________________________________________

# abrir imagem
def open_img():
    filename = filedialog.askopenfilename(parent=root, filetypes=[("PNG", ".png"), ("JPG", ".jpg")])
    if filename:
        global file_path, no_loaded_text_exists, loaded_img

        file_path = filename
                
        if no_loaded_text_exists:
            m_frame.nametowidget("no_loaded_img").destroy()
            no_loaded_text_exists = False

        img = Image.open(filename)
        loaded_img = img
        width, height = img.size

        img = img.resize((width // 2, height // 2))
        img_component = ImageTk.PhotoImage(img)
        panel = Label(m_frame, name="loaded_img", image=img_component)
        panel.image = img_component
        panel.grid(column=0, row=0)

        is_img_loaded = True
        enable_buttons()
#______________________________________________

#habilitar e desabilitar botões
def enable_buttons():
    b_frame.nametowidget("b1")["state"] = "normal"
    b_frame.nametowidget("b2")["state"] = "normal"
    filemenu.entryconfig("Ampliar (Zoom In)", state="normal")
    filemenu.entryconfig("Reduzir (Zoom Out)", state="normal")

def disable_buttons():
    b_frame.nametowidget("b1")["state"] = "disabled"
    b_frame.nametowidget("b2")["state"] = "disabled"
    filemenu.entryconfig("Ampliar (Zoom In)", state="disabled")
    filemenu.entryconfig("Reduzir (Zoom Out)", state="disabled")
#______________________________________________

#aumentar e diminuir o zoom da imagem
def zoomIn_img():
    label = m_frame.nametowidget("loaded_img")
    scale_w = label.image.width() * 2
    scale_h = label.image.height() * 2
    
    img_component = ImageTk.PhotoImage(loaded_img.resize((scale_w, scale_h)))
    label["image"] = img_component
    label.image = img_component

def zoomOut_img():
    label = m_frame.nametowidget("loaded_img")
    scale_w = label.image.width() // 2
    scale_h = label.image.height() // 2
    
    img_component = ImageTk.PhotoImage(loaded_img.resize((scale_w, scale_h)))
    label["image"] = img_component
    label.image = img_component
#______________________________________________

#manipulação de histograma
def convert_to_gray():
    #converte a imagem carregada para escala de cinza
    params = [0.2125, 0.7154, 0.0721] 
    n_image = np.array(loaded_img)
    gray_image = np.ceil(np.dot(n_image[..., :3], params))
    gray_image[gray_image > 255] = 255
    return gray_image

def create_histogram():    
    #cria dicionário para o histograma
    hist_dct = {str(i): 0 for i in range(256)} 
    return hist_dct

def count_intensity_values(hist, img):
    #conta os valores de intensidade da imagem
    for row in img:
        for column in row:
            hist[str(int(column))] += 1
    return hist

def display_hist():
    #exibe o histograma da imagem em escala de cinza
    gray_img = convert_to_gray()
    hist = create_histogram()
    hist = count_intensity_values(hist, gray_img)

    newWindow = Toplevel(root)
    newWindow.title("Histograma tons de cinza")
    newWindow.geometry("512x512")

    plt.bar(hist.keys(), hist.values())
    plt.xlabel("Níveis de intensidade")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    plt.grid(True)

    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=newWindow)   
    canvas.draw() 
    canvas.get_tk_widget().pack() 
#______________________________________________

#recorte do ROI
def crop_ROI():  
    left = float(roi_menu.nametowidget("left").get())
    upper = float(roi_menu.nametowidget("upper").get())
    right = float(roi_menu.nametowidget("right").get())
    down = float(roi_menu.nametowidget("down").get())

    fileName = roi_menu.nametowidget("roi_fileName").get()
    cropped_img = loaded_img.crop((left, upper, right, down))
    cropped_img.save(fileName + ".png")

def open_ROI_window():
    global roi_menu

    newWindow = Toplevel(root, name="roi_menu")
    newWindow.title("Recortar ROI")
    newWindow.geometry("400x320")

    tmp_frame = Frame(newWindow)
    roi_menu = tmp_frame
    
    Label(tmp_frame, text="Esquerda").pack()
    Entry(tmp_frame, name="left").pack()
    Label(tmp_frame, text="Cima").pack()
    Entry(tmp_frame, name="upper").pack()
    Label(tmp_frame, text="Direita").pack()
    Entry(tmp_frame, name="right").pack()
    Label(tmp_frame, text="Baixo").pack()
    Entry(tmp_frame, name="down").pack(pady=5)

    s1 = ttk.Separator(tmp_frame, orient='horizontal')
    s1.pack(fill='x', pady=5)
    
    Label(tmp_frame, text="Nome do ROI").pack()
    Entry(tmp_frame, name="roi_fileName").pack(pady=3)
    Button(tmp_frame, text="Recortar", command=crop_ROI).pack(pady=5)

    s2 = ttk.Separator(tmp_frame, orient='horizontal')
    s2.pack(fill='x', pady=5)
    
    Label(tmp_frame, text="Manual: Passe as coordenadas da caixa na qual você deseja recortar").pack()
    tmp_frame.pack()
#______________________________________________

#janela da aplicação
root = Tk()
root.title("Detector de NAFLD") 
root.geometry('854x480')
root.maxsize(854, 480)

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.rowconfigure(0, weight=2)

#______________________________________________

#exibe a imagem a partir de um array
def display_image_from_array(img_array):
    img = Image.fromarray(img_array)
    img_component = ImageTk.PhotoImage(img)
    panel = Label(m_frame, image=img_component)
    panel.image = img_component
    panel.grid(column=0, row=0)
#______________________________________________

#arquivo .mat
def open_mat_file():
    filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    if filename:
        data = scipy.io.loadmat(filename)
        images = data['images']  # Acessa as imagens
        display_image_from_array(images[0][0])  # Exibe a primeira imagem
#______________________________________________

#menu 
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)

filemenu.add_command(label="Carregar imagem", command=open_img)
filemenu.add_separator()
filemenu.add_command(label="Ampliar (Zoom In)", command=zoomIn_img, state="disabled")
filemenu.add_command(label="Reduzir (Zoom Out)", command=zoomOut_img, state="disabled")
filemenu.add_separator()
filemenu.add_command(label="Carregar arquivo MAT", command=open_mat_file)
filemenu.add_command(label="Sair", command=root.quit)
menubar.add_cascade(label="Menu", menu=filemenu)

root.config(menu=menubar)
#______________________________________________

#botões
b_frame = Frame(root)
b_frame.grid(column=0, row=0)

Button(b_frame, name="b1", text="Recortar ROI", width=15, cursor="hand2", command=open_ROI_window, state="disabled").grid(column=0, row=0, pady=2)
Button(b_frame, name="b2", text="Histograma", width=15, cursor="hand2", command=display_hist, state="disabled").grid(column=0, row=1, pady=2)
#______________________________________________

#frame exibição da imagem
m_frame = Frame(root, bg="white", highlightbackground="gray", highlightthickness=1, width=640, height=360)
m_frame.grid_propagate(False)
m_frame.grid(column=1, row=0)

Label(m_frame, name="no_loaded_img", text="Nenhuma imagem carregada", bg="white").grid(column=0, row=0)

root.mainloop()

