import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal

from testing.runner import run_benchmark



class TestingPage(QWidget):
    back_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self._updating_buses = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FDIA Testing Module")
        self.setMinimumSize(600, 750)

        main_layout = QVBoxLayout()

        # ---------------- File Selection ----------------
        main_layout.addWidget(QLabel("Select Real (Authentic) Dataset File"))
        self.real_file_btn = QPushButton("Select Real Data File")
        self.real_file_btn.clicked.connect(self.select_real_file)
        main_layout.addWidget(self.real_file_btn)

        self.real_file_label = QLabel("No file selected")
        main_layout.addWidget(self.real_file_label)

        main_layout.addWidget(QLabel("Select FDIA Dataset File"))
        self.fdia_file_btn = QPushButton("Select FDIA File")
        self.fdia_file_btn.clicked.connect(self.select_fdia_file)
        main_layout.addWidget(self.fdia_file_btn)

        self.fdia_file_label = QLabel("No file selected")
        main_layout.addWidget(self.fdia_file_label)

        # ---------------- Train/Test Split ----------------
        main_layout.addWidget(QLabel("Training Data Percentage (e.g. 70)"))

        self.train_percent = QLineEdit()
        self.train_percent.setPlaceholderText("Enter training percentage (0–100)")
        main_layout.addWidget(self.train_percent)

        main_layout.addWidget(QLabel("Testing Data Percentage (e.g. 30)"))

        self.test_percent = QLineEdit()
        self.test_percent.setPlaceholderText("Enter testing percentage (0–100)")
        main_layout.addWidget(self.test_percent)

        # ---------------- Algorithm Selection ----------------
        main_layout.addWidget(QLabel("Select ML / DL Algorithms"))

        self.algo_list = QListWidget()
        self.algo_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        algorithms = [
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "MLP"
        ]

        for algo in algorithms:
            item = QListWidgetItem(algo)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.algo_list.addItem(item)

        main_layout.addWidget(self.algo_list)

        # ---------------- Column Selection ----------------
        main_layout.addWidget(QLabel("Select Columns (Buses)"))

        self.bus_list = QListWidget()
        self.bus_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        select_all = QListWidgetItem("Select All Buses")
        select_all.setCheckState(Qt.CheckState.Unchecked)
        self.bus_list.addItem(select_all)

        for i in range(1, 119):
            item = QListWidgetItem(f"Bus {i}")
            item.setCheckState(Qt.CheckState.Unchecked)
            self.bus_list.addItem(item)

        self.bus_list.itemChanged.connect(self.handle_select_all)
        main_layout.addWidget(self.bus_list)

        # ---------------- Metrics Selection ----------------
        main_layout.addWidget(QLabel("Select Evaluation Metrics"))

        self.metric_list = QListWidget()
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

        for metric in metrics:
            item = QListWidgetItem(metric)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.metric_list.addItem(item)

        main_layout.addWidget(self.metric_list)

        # ---------------- Generate ----------------
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate_clicked)
        main_layout.addWidget(self.generate_btn)

        self.back_btn = QPushButton("Back to FDIA Generation")
        self.back_btn.clicked.connect(self.back_requested.emit)
        main_layout.addWidget(self.back_btn)


        self.setLayout(main_layout)

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------

    def select_real_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Real Dataset")
        if file:
            self.real_file_label.setText(file)

    def select_fdia_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select FDIA Dataset")
        if file:
            self.fdia_file_label.setText(file)

    # def handle_select_all(self, item):
    #     if item.text() == "Select All Buses":
    #         state = item.checkState()
    #         for i in range(1, self.bus_list.count()):
    #             self.bus_list.item(i).setCheckState(state)

    def handle_select_all(self, item):
        if self._updating_buses:
            return

        self._updating_buses = True

        try:
            if item.text() == "Select All Buses":
                state = item.checkState()
                # Apply state to all bus items (skip index 0)
                for i in range(1, self.bus_list.count()):
                    self.bus_list.item(i).setCheckState(state)
            else:
            # If any individual bus is unchecked, uncheck "Select All"
                all_checked = True
                for i in range(1, self.bus_list.count()):
                    if self.bus_list.item(i).checkState() != Qt.CheckState.Checked:
                        all_checked = False
                        break

                select_all_item = self.bus_list.item(0)
                select_all_item.setCheckState(
                    Qt.CheckState.Checked if all_checked else Qt.CheckState.Unchecked
                )
        finally:
            self._updating_buses = False


    def generate_clicked(self):
        real_file = self.real_file_label.text()
        fdia_file = self.fdia_file_label.text()

        # Basic validation for file selection buttons
        if not real_file or real_file == "No file selected":
            QMessageBox.warning(self, "Missing input", "Please select the Real (Authentic) dataset CSV file.")
            return
        if not fdia_file or fdia_file == "No file selected":
            QMessageBox.warning(self, "Missing input", "Please select the FDIA dataset CSV file.")
            return

        train_pct = self.train_percent.text().strip()
        test_pct = self.test_percent.text().strip()

        try:
            train_pct_f = float(train_pct)
            test_pct_f = float(test_pct)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Train/Test percentages must be numbers (e.g., 70 and 30).")
            return

        selected_algos = [
            self.algo_list.item(i).text()
            for i in range(self.algo_list.count())
            if self.algo_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        selected_buses = [
            self.bus_list.item(i).text()
            for i in range(1, self.bus_list.count())
            if self.bus_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        selected_metrics = [
            self.metric_list.item(i).text()
            for i in range(self.metric_list.count())
            if self.metric_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        config = {
            "real_file": real_file,
            "fdia_file": fdia_file,
            "train_pct": train_pct_f,
            "test_pct": test_pct_f,
            "selected_algos": selected_algos,
            "selected_buses": selected_buses,
            "selected_metrics": selected_metrics,
        }

        try:
            results = run_benchmark(config)
        except Exception as e:
            QMessageBox.critical(self, "Benchmark failed", str(e))
            return

        # Build a simple readable summary for a popup
        lines = [
            f"Train samples: {results['n_train']}",
            f"Test samples: {results['n_test']}",
            f"Features used: {results['n_features']}",
            "",
        ]

        for algo_name, payload in results["algorithms"].items():
            lines.append(f"[{algo_name}]")
            metrics = payload.get("metrics", {})
            if not metrics:
                lines.append("  (No metrics selected)")
            else:
                for k, v in metrics.items():
                    lines.append(f"  {k}: {v}")
            lines.append("")

        QMessageBox.information(self, "Benchmark Results", "\n".join(lines))


# -------------------------------------------------
# Run Standalone
# -------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestingPage()
    window.show()
    sys.exit(app.exec())
