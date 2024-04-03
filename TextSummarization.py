import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QComboBox, QPushButton, QTextEdit)

class HoverButton(QPushButton):
    def __init__(self, text, parent=None):
        super(HoverButton, self).__init__(text, parent)
        self.setStyleSheet("QPushButton { background-color: #50727B; color: white; border-radius: 5px; }"
                           "QPushButton:hover { background-color: #6299A8; }")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text Summarization")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: #344955;")

        layout = QVBoxLayout()

        layout.addWidget(QLabel("<center><font color='white'><h1>Text Summarization</h1></font></center>"))

        self.input_text_edit = QTextEdit()
        layout.addWidget(QLabel("<font color='white'>Input Text</font>"))
        layout.addWidget(self.input_text_edit)
        self.input_text_edit.setStyleSheet("background-color: #E3E1D9; color: black;")

        self.output_text_edit = QTextEdit()
        layout.addWidget(QLabel("<font color='white'>Summary</font>"))
        layout.addWidget(self.output_text_edit)
        self.output_text_edit.setStyleSheet("background-color: #E3E1D9; color: black;")


        self.level_combo_box = QComboBox()
        self.level_combo_box.addItems(["Low", "Medium", "High"])
        self.level_combo_box.setFixedSize(200, 30)
        self.level_combo_box.setStyleSheet("color: white; background-color: #78A083; border-radius: 5px;")
        layout.addWidget(QLabel("<font color='white'>Summary Level</font>"))
        layout.addWidget(self.level_combo_box)


        self.summarize_button = HoverButton("Summarize")
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summarize_button.setFixedSize(400, 40)
        layout.addWidget(self.summarize_button)

        self.central_widget.setLayout(layout)


    def summarize_text(self):
        text = self.input_text_edit.toPlainText()
        text = text.replace('\n', '')
        sentence = re.split('\. |\.', text)

        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokenized = [tokenizer.tokenize(s.lower()) for s in sentence]

        listStopword = set(stopwords.words('indonesian'))

        important_token = []
        for sent in tokenized:
            filtered = [s for s in sent if s not in listStopword]
            important_token.append(filtered)

        sw_removed = [' '.join(t) for t in important_token]

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stemmed_sent = [stemmer.stem(sent) for sent in sw_removed]

        vec = TfidfVectorizer(lowercase=True)
        document = vec.fit_transform(stemmed_sent)

        document = document.toarray()

        summary_levels = {"Low": 12, "Medium": 10, "High": 8}
        n = summary_levels[self.level_combo_box.currentText()]

        result = np.sum(document, axis=1).flatten() 

        top_n = np.argsort(result)[-n:]
        summ_index = sorted(top_n)

        summary = ''
        for i in summ_index:
            summary += sentence[i] + '.\n'

        self.output_text_edit.setPlainText(summary)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
