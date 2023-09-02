import cv2 as cv
import numpy as np
import random as rnd
import sys

from matplotlib import pyplot as plt
from matplotlib.widgets import Button as btn
from scipy.spatial.distance import cosine as csn
from scipy.ndimage import gaussian_gradient_magnitude as ggm
from skimage.metrics import structural_similarity as ssim
from PyQt5 import QtCore, QtGui, QtWidgets

# ----------------------------------------------------------------------------

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(227, 78)
        MainWindow.setMinimumSize(QtCore.QSize(227, 78))
        MainWindow.setMaximumSize(QtCore.QSize(227, 78))
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(9, 9, 148, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(163, 9, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(9, 35, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.button_click)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа"))
        self.label.setText(_translate("MainWindow", "Введите кол-во человек:"))
        self.pushButton.setText(_translate("MainWindow", "Принять"))
    
    def button_click(self):
        Program(int(self.lineEdit.text())).show_result()
        
class Sketch():
    def __init__(self, i, tp):
        self.tp = tp
        self.sketch = cv.imread(f"cuhk\sketches\{i}.jpg")
        self.sketch_gray = cv.imread(f"cuhk\sketches\{i}.jpg", 0)
        
    def get_sketch(self):
        if self.tp == 'normal':
            return self.sketch
        
        elif self.tp == 'gray':
            return self.sketch_gray

class Photo():
    def __init__(self, i, tp):
        self.tp = tp
        self.photo = cv.imread(f"cuhk\photos\{i}.jpg")
        self.photo_gray = cv.imread(f"cuhk\photos\{i}.jpg", 0)
        
    def get_photo(self):
        if self.tp == 'normal':
            return self.photo
        
        elif self.tp == 'gray':
            return self.photo_gray
            
# ----------------------------------------------------------------------------

class Program():
    def __init__(self, count):
        self.photo_dataset = []
        self.sketch_dataset = []
        self.count = count
        self.get_dataset()
    
    def get_dataset(self):
        for i in range(188):
            sketch = Sketch(i + 1, 'gray').get_sketch()
            self.sketch_dataset.append(sketch)
            
        for i in range(self.count):
            photo = Photo(i + 1, 'gray').get_photo()
            self.photo_dataset.append(photo)
    
    def create_populations(self, count):
        self.sketch_populations = [[] for i in range(len(self.sketch_dataset))]
        
        for i in range(len(self.sketch_dataset)):
            p1 = self.sketch_dataset[i].shape[0]
            p2 = self.sketch_dataset[i].shape[1]
            
            rnd1 = rnd.randint(5, 25)
            rnd2 = rnd.randint(5, 25)
            rnd3 = rnd.randint(-15, 15)
            
            if rnd3 < 0:
                x = int(rnd1 / 2)
                y = x
                z = int(rnd2 / 2 + rnd3)
                
            elif rnd3 > 0:
                x = int(rnd1 / 2)
                y = int(rnd2 / 2 + rnd3)
                z = x

            else:
                x = int(rnd1 / 2)
                y = int(rnd2 / 2)
                z = y
            
            sketch = self.sketch_dataset[i][x:p1 - x, y:p2 - z]
            
            new_sketch1 = cv.resize(sketch, (p2, p1))
            new_sketch2 = np.mean([self.sketch_dataset[i], new_sketch1], 
                                  axis=0)
            
            self.sketch_populations[i].append(new_sketch1)
            self.sketch_populations[i].append(new_sketch2)

    def dct(self, photo, sketch):
        dct1 = cv.dct(np.float32(photo))
        dct2 = cv.dct(np.float32(sketch))

        sim = 1 - csn(dct1.flatten(), dct2.flatten())

        return sim * 100
    
    def dft(self, photo, sketch):
        dft1 = cv.dft(np.float32(photo), flags=cv.DFT_COMPLEX_OUTPUT)
        dft2 = cv.dft(np.float32(sketch), flags=cv.DFT_COMPLEX_OUTPUT)
        
        dist = np.linalg.norm(dft1 - dft2)
        max_dist = np.sqrt(photo.shape[0] * photo.shape[1] * 2) * 255

        return (max_dist / dist) * 1000
    
    def compare_images(self, count):
        self.result = [[] for i in range(count)]
        
        for i in range(count):
            compared = []
            
            for j in range(len(self.sketch_dataset)):
                compared.append(self.dct(self.photo_dataset[i],
                                              self.sketch_dataset[j]))
                
            sim1 = ssim(self.photo_dataset[i], self.sketch_populations[i][0], data_range=1.0)
            sim2 = ssim(self.photo_dataset[i], self.sketch_populations[i][1], data_range=1.0)
            
            self.result[i] = [np.argmax(compared), np.max(compared),
                              sim1, sim2]
    
    def get_accuracy(self, count):
        self.accuracy = [[] for i in range(count)]
        
        trues = 0
        alls = 0

        for i in range(count):
            alls += 1

            if self.result[i][0] == i:
                trues += 1

            self.accuracy[i].append((trues / alls) * 100)
    
    def show_result(self):
        self.create_populations(self.count)
        self.compare_images(self.count)
        self.get_accuracy(self.count)
        
        self.flag = True
        
        def stop(event):
            self.flag = not self.flag
        
        fig = plt.figure('Результат', figsize=(16, 8))
        
        ax1 = fig.add_subplot(2, 4, 1)
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 5)
        ax4 = fig.add_subplot(2, 4, 6)
        ax5 = fig.add_subplot(1, 4, 3)
        ax6 = fig.add_subplot(1, 4, 4)
        
        ax7 = plt.axes([0.85, 0.05, 0.1, 0.05])
        button = btn(ax7, 'Остановить', color='gray')
        button.on_clicked(stop)
        
        y = [[], []]
        x = []
        z = []
        
        for i in range(self.count):
            y[0].append(self.result[i][2])
            y[1].append(self.result[i][3])
            x.append(i + 1)
            z.append(self.accuracy[i])
            
            ax1.cla()
            ax1.imshow(Photo(i + 1, 'gray').get_photo(), cmap='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('Фото')
            ax1.set_xlabel('Класс = ' + str(i + 1))
            
            ax2.cla()
            ax2.imshow(Sketch(self.result[i][0] + 1, 'gray').get_sketch(), 
                       cmap='gray')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title('Скетч')
            ax2.set_xlabel('Класс = ' + str(self.result[i][0] + 1) + '\nСходство с фото = ' +
                           str(round(self.result[i][1], 1)) + '%')
            
            ax3.cla()
            ax3.imshow(self.sketch_populations[self.result[i][0]][0], 
                       cmap='gray')
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('Популяция 1')
            
            ax4.cla()
            ax4.imshow(self.sketch_populations[self.result[i][0]][1], 
                       cmap='gray')
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_title('Популяция 2')
            
            ax5.cla()
            ax5.plot(x, y[0], label='П1')
            ax5.plot(x, y[1], label='П2')
            ax5.set_xlabel('Кол-во человек')
            ax5.set_ylabel('SSIM')
            ax5.set_title('Индекс структурного сходства')
            ax5.legend()
            
            ax6.cla()
            ax6.plot(x, z)
            ax6.set_xlabel('Кол-во человек')
            ax6.set_ylabel('Точность (%)')
            ax6.set_title('Точность распознавания')
            ax6.set_yticks(np.arange(0, 110, 10))
            
            plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.95,
                                bottom=0.2, left=0.01, right=0.96)
            plt.show()
            plt.pause(0.01)
            
            if not self.flag:                  
                break

# ----------------------------------------------------------------------------

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())