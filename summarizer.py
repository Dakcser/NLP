import sys
from PySide2.QtCore import Qt, Slot
from PySide2.QtWidgets import (QAction, QApplication, QHeaderView, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QPushButton,
                               QTableWidget, QTableWidgetItem, QDoubleSpinBox,
                               QVBoxLayout, QWidget, QGroupBox, QCheckBox,
                               QSpinBox, QFormLayout, QRadioButton)

class Widget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Store input document
        self.document = None

        ################### UI Elements ####################
        self.file_name = QLineEdit()
        self.load = QPushButton("Load document")
        self.load.setStyleSheet("background-color:#eb8c34;")

        table = QTableWidget()

        quit = QPushButton("Quit")
        quit.setStyleSheet("background-color:#eb8c34;")

        # Disabling 'Load' button
        self.load.setEnabled(False)

        # Load file VBox
        loadfile = QVBoxLayout()
        loadfile.setMargin(10)
        loadfile.addWidget(QLabel("Document location (URL or file path)"))
        loadfile.addWidget(self.file_name)
        loadfile.addWidget(self.load)

        # Box for radiobuttons
        radiobuttonsBox = QGroupBox("Choose where the given document is located before loading the document")
        radiobuttonsBox.setFlat(True)

        self.url = QRadioButton("&URL")
        self.path = QRadioButton("&File path")
        # Set url option to true by default
        self.url.setChecked(True)

        radiobuttonsLayout = QVBoxLayout()
        radiobuttonsLayout.addWidget(self.url)
        radiobuttonsLayout.addWidget(self.path)
        radiobuttonsBox.setLayout(radiobuttonsLayout)

        # Element containing the loading options
        loadview = QHBoxLayout()
        loadview.addLayout(loadfile)
        loadview.addWidget(radiobuttonsBox)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(loadview)
        layout.addWidget(table)
        layout.addWidget(quit)


        # Set the layout to the QWidget
        self.setLayout(layout)

        # Connect buttons with functions
        quit.clicked.connect(self.quit_application)
        self.file_name.textChanged[str].connect(self.check_disable)
        self.load.clicked.connect(self.load_document)

    @Slot()
    def check_disable(self, s):
        """
        Disable load data button if no path is given.
        """
        if not self.file_name.text():
            self.load.setEnabled(False)
        else:
            self.load.setEnabled(True)

    @Slot()
    def load_document(self):
        """
        Load data from given path
        """
        try:
            document = None
            # Document location is given by url
            if self.url.isChecked():
                pass
            # Document is stored locally
            else:
                ftype = self.file_name.text().split('.')[-1]
                if ftype == 'html' or ftype == 'txt':
                    pass
                else:
                    raise FileNotFoundError("File must be html or txt document")
            
            self.document = document
        except FileNotFoundError:
            self.dialog = FileNotFoundWindow()
            self.dialog.show()
 
    @Slot()
    def quit_application(self):
        """
        Quit application.
        """
        QApplication.quit()


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("Automatic Summarizer")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self, checked):
        QApplication.quit()


class FileNotFoundWindow(QMainWindow):
    def __init__(self, parent=None):
        super(FileNotFoundWindow, self).__init__(parent)
        self.setWindowTitle("Error")

        # Display error message
        self.message = QLabel(self)
        self.message.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.message.setText("File not found or it is in wrong format!\nSupported types are .html and .txt.\nIf file is in the same folder with this code just type the file name.\nFor example file.html.")

        # Push button for closing window
        self.ok = QPushButton("Ok")
        self.ok.setStyleSheet("background-color:#eb8c34;")

        # Page layout
        layout = QVBoxLayout()
        layout.addWidget(self.message)
        layout.addWidget(self.ok)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.ok.clicked.connect(self.close_window)

    @Slot()
    def close_window(self):
        self.close()


if __name__ == "__main__":
    # Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # QWidget
    widget = Widget()
    # QMainWindow using QWidget as central widget
    window = MainWindow(widget)
    window.resize(1000, 800)
    window.show()

    # Execute application
    sys.exit(app.exec_())