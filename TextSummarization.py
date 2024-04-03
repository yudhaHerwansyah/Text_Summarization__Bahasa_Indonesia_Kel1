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

        # Calculate accuracy
        reference_summary = """sriwijaya adalah kerajaan bahari historis yang berasal dari Pulau Sumatra sekitar abad ke-7 sampai abad ke-11.
Kehadirannya banyak memberi pengaruh pada perkembangan sejarah Asia Tenggara (terutama dalam kawasan Nusantara barat).
Dalam bahasa Sanskerta, sri berarti "bercahaya" atau "gemilang", dan vijaya berarti "kemenangan" atau "kejayaan".
Lokasi ibukota Sriwijaya dapat dengan akurat disimpulkan berada di Kota Palembang, tepatnya di muara Sungai Musi.
Bukti awal mengenai keberadaan kerajaan ini berasal dari abad ke-7; seorang pendeta Tiongkok dari Dinasti Tang, I Tsing, menulis bahwa ia mengunjungi Sriwijaya tahun 671 dan tinggal selama 6 bulan.
Meskipun sempat dianggap sebagai talasokrasi (kerajaan berbasis maritim), penelitian baru tentang catatan yang tersedia menunjukkan bahwa Sriwijaya merupakan negara berbasis darat daripada kekuatan maritim.
Armada laut memang tersedia tetapi bertindak sebagai dukungan logistik untuk memfasilitasi proyeksi kekuatan darat.
Menanggapi perubahan ekonomi maritim Asia, dan terancam oleh hilangnya negara bawahannya, kerajaan-kerajaan disekitar selat Malaka mengembangkan strategi angkatan laut untuk menunda kemerosotannya.
Strategi angkatan laut kerajaan-kerajaan disekitar selat Malaka bersifat menghukum untuk memaksa kapal-kapal dagang datang ke pelabuhan mereka.
Setelah itu, kerajaan ini terlupakan dan keberadaannya baru diketahui kembali lewat publikasi tahun 1918 oleh sejarawan Prancis George Cœdès dari École française d'Extrême-Orient."""
        
        generated_summary = summary.split('.\n')
        reference_summary = reference_summary.split('.\n')

        correct_sentences = sum(1 for sent in generated_summary if sent in reference_summary)
        total_sentences = len(reference_summary)

        accuracy = correct_sentences / total_sentences * 100
        print(f"Accuracy: {accuracy}%")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()