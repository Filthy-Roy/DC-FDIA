import sys
import random
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QMessageBox
)
from PyQt6.QtCore import Qt
from logic import generate_fdia
from PyQt6.QtCore import pyqtSignal



class FDIAFrontend(QWidget):
    test_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FDIA Dataset Generator")
        self.setMinimumSize(600, 700)

        main_layout = QVBoxLayout()

        # ---------------- Dataset Size ----------------
        main_layout.addWidget(QLabel("Enter The Size of the Dataset (Max 70,000)"))

        self.dataset_size_input = QLineEdit()
        self.dataset_size_input.setPlaceholderText("e.g. 5000, 10000, 15000")
        main_layout.addWidget(self.dataset_size_input)

        # ---------------- Attack Area ----------------
        main_layout.addWidget(QLabel("Select Attack Area (Bus Numbers)"))

        self.bus_list = QListWidget()
        self.bus_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        for i in range(1, 119):
            item = QListWidgetItem(f"Bus {i}")
            item.setCheckState(Qt.CheckState.Unchecked)
            self.bus_list.addItem(item)

        main_layout.addWidget(self.bus_list)

        # ---------------- Attack Magnitude ----------------
        main_layout.addWidget(QLabel("Enter Attack Magnitude (0–100)"))

        mag_layout = QHBoxLayout()
        self.mag_from = QLineEdit()
        self.mag_to = QLineEdit()
        self.mag_from.setPlaceholderText("From")
        self.mag_to.setPlaceholderText("To")
        mag_layout.addWidget(self.mag_from)
        mag_layout.addWidget(self.mag_to)

        main_layout.addLayout(mag_layout)

        # ---------------- Meter Range ----------------
        main_layout.addWidget(QLabel("Enter Number of Meters to Select Randomly"))

        meter_layout = QHBoxLayout()
        self.meter_from = QLineEdit()
        self.meter_to = QLineEdit()
        self.meter_from.setPlaceholderText("From")
        self.meter_to.setPlaceholderText("To")
        meter_layout.addWidget(self.meter_from)
        meter_layout.addWidget(self.meter_to)

        main_layout.addLayout(meter_layout)

        # ---------------- Dataset Names ----------------
        main_layout.addWidget(QLabel("Enter Dataset Names"))

        name_layout = QHBoxLayout()
        self.auth_name = QLineEdit()
        self.fdia_name = QLineEdit()
        self.auth_name.setPlaceholderText("Authentic Dataset Name")
        self.fdia_name.setPlaceholderText("FDIA Dataset Name")
        name_layout.addWidget(self.auth_name)
        name_layout.addWidget(self.fdia_name)

        main_layout.addLayout(name_layout)

        # ---------------- Buttons ----------------
        button_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate")
        self.test_btn = QPushButton("Test Generated Data")

        self.generate_btn.clicked.connect(self.generate_clicked)
        self.test_btn.clicked.connect(self.test_clicked)

        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.test_btn)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    # -------------------------------------------------
    # Button Actions
    # -------------------------------------------------

    def generate_clicked(self):
        try:
            dataset_size = int(self.dataset_size_input.text())
            if dataset_size <= 0 or dataset_size > 70000:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Dataset size must be 1–70,000")
            return

        selected_buses = [
            i + 1
            for i in range(self.bus_list.count())
            if self.bus_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        if not selected_buses:
            QMessageBox.warning(
            self,
                "No Attack Area Selected",
                "Please select at least one bus for the attack area."
            )
            return
        try:
            mag_from = float(self.mag_from.text())
            mag_to = float(self.mag_to.text())
            mag_from /= 100
            mag_to /= 100
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Attack magnitude must be numeric")
            return

        try:
            meter_from = int(self.meter_from.text())
            meter_to = int(self.meter_to.text())
            selected_meter_count = random.randint(meter_from, meter_to)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Meter range must be numeric")
            return

        auth_name = self.auth_name.text().strip()
        fdia_name = self.fdia_name.text().strip()

        if not auth_name or not fdia_name:
            QMessageBox.warning(
                self,
                "Missing Dataset Name",
                "Please enter BOTH authentic and FDIA dataset names."
            )
            return


        # # ---------------- PRINT TO CONSOLE ----------------
        # print("\n===== FDIA PARAMETERS =====")
        # print(f"Dataset Size: {dataset_size}")
        # print(f"Selected Buses: {selected_buses}")
        # print(f"Attack Magnitude Range: {mag_from} – {mag_to}")
        # print(f"Meters Selected Per Row: {selected_meter_count}")
        # print(f"Authentic Dataset Name: {auth_name}")
        # print(f"FDIA Dataset Name: {fdia_name}")
        # print("===========================\n")


        try:
            stealth_count  = generate_fdia(
                dataset_size=dataset_size,
                buses=selected_buses,
                attack_mag_range=(mag_from, mag_to),
                meter_range=(meter_from, meter_to),
                authentic_name=auth_name,
                fdia_name=fdia_name
            )   
            if stealth_count is not None:
                QMessageBox.information(
                    self,
                    "Success",
                    f"FDIA dataset generated successfully!\nStealth Count from residual test: {stealth_count}"
                )
        except Exception as e:
            QMessageBox.critical(
                    self,
                    "Generation Failed",
                    f"An error occurred:\n{str(e)}"
                )
    def test_clicked(self):
        self.test_requested.emit()



# -------------------------------------------------
# App Entry Point
# -------------------------------------------------

