import os
import csv
import sqlite3
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance_system.db")
FACES_DIR = os.path.join(BASE_DIR, "data", "faces")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
CAMERA_INDEX = 0
TOLERANCE = 0.48
REQUIRED_SAMPLES = 5

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT DEFAULT '',
                year TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                encoding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                UNIQUE(student_id, attendance_date),
                FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def add_student(self, student_id, name, department, year, encodings):
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO students VALUES (?, ?, ?, ?, ?)",
                (student_id, name, department, year, now)
            )
            for encoding in encodings:
                self.conn.execute(
                    "INSERT INTO face_encodings(student_id, encoding, created_at) VALUES (?, ?, ?)",
                    (student_id, encoding.astype(np.float64).tobytes(), now)
                )

    def students(self):
        return self.conn.execute(
            "SELECT student_id, name, department, year FROM students ORDER BY name"
        ).fetchall()

    def encodings(self):
        rows = self.conn.execute(
            "SELECT student_id, encoding FROM face_encodings"
        ).fetchall()
        ids, vectors = [], []
        for student_id, blob in rows:
            ids.append(student_id)
            vectors.append(np.frombuffer(blob, dtype=np.float64))
        return ids, vectors

    def mark_present(self, student_id):
        now = datetime.now()
        try:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO attendance(student_id, attendance_date, attendance_time, status)
                       VALUES (?, ?, ?, 'Present')""",
                    (student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def attendance(self, start=None, end=None):
        sql = """SELECT a.student_id, s.name, a.attendance_date, a.attendance_time, a.status
                 FROM attendance a JOIN students s ON s.student_id=a.student_id"""
        params = []
        if start and end:
            sql += " WHERE a.attendance_date BETWEEN ? AND ?"
            params = [start, end]
        sql += " ORDER BY a.attendance_date DESC, a.attendance_time DESC"
        return self.conn.execute(sql, params).fetchall()

    def delete_student(self, student_id):
        with self.conn:
            self.conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))

    def close(self):
        self.conn.close()


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance | Student Management")
        self.root.geometry("1120x720")
        self.root.minsize(950, 620)
        self.db = Database()
        self.camera = None
        self.running = False
        self.known_ids, self.known_encodings = self.db.encodings()
        self.configure_style()
        self.build_ui()
        self.refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_style(self):
        self.root.configure(bg="#eef2f7")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", foreground="#172033", font=("Arial", 10))
        style.configure("Title.TLabel", font=("Arial", 22, "bold"), foreground="#152746")
        style.configure("Sub.TLabel", font=("Arial", 10), foreground="#667085")
        style.configure("TButton", font=("Arial", 10, "bold"), padding=9)
        style.configure("Treeview", rowheight=29, font=("Arial", 10))
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

    def build_ui(self):
        header = ttk.Frame(self.root, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="FACE ATTENDANCE SYSTEM", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="By Nitin Kachap", style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.dashboard_tab = ttk.Frame(self.tabs, padding=18)
        self.register_tab = ttk.Frame(self.tabs, padding=18)
        self.attendance_tab = ttk.Frame(self.tabs, padding=18)
        self.records_tab = ttk.Frame(self.tabs, padding=18)
        self.tabs.add(self.dashboard_tab, text="Dashboard")
        self.tabs.add(self.register_tab, text="Register Student")
        self.tabs.add(self.attendance_tab, text="Live Attendance")
        self.tabs.add(self.records_tab, text="Attendance Records")
        self.build_dashboard()
        self.build_register()
        self.build_attendance()
        self.build_records()

    def build_dashboard(self):
        ttk.Label(self.dashboard_tab, text="Overview", style="Title.TLabel").pack(anchor="w", pady=(0, 16))
        cards = ttk.Frame(self.dashboard_tab)
        cards.pack(fill="x")
        self.student_count = tk.StringVar(value="0")
        self.today_count = tk.StringVar(value="0")
        self.total_count = tk.StringVar(value="0")
        for title, var in [("REGISTERED STUDENTS", self.student_count),
                           ("PRESENT TODAY", self.today_count),
                           ("TOTAL ATTENDANCE LOGS", self.total_count)]:
            card = ttk.LabelFrame(cards, text=title, padding=18)
            card.pack(side="left", fill="both", expand=True, padx=(0, 12))
            ttk.Label(card, textvariable=var, font=("Arial", 26, "bold"), foreground="#1769aa").pack(anchor="w")
        ttk.Label(self.dashboard_tab, text="Quick actions", font=("Arial", 14, "bold")).pack(anchor="w", pady=(28, 12))
        actions = ttk.Frame(self.dashboard_tab)
        actions.pack(anchor="w")
        ttk.Button(actions, text="Register a student", command=lambda: self.tabs.select(self.register_tab)).pack(side="left", padx=(0, 10))
        ttk.Button(actions, text="Start live attendance", command=lambda: self.tabs.select(self.attendance_tab)).pack(side="left")
        ttk.Label(self.dashboard_tab, text="Tip: Register each student with clear, front-facing camera samples before starting attendance.", style="Sub.TLabel").pack(anchor="w", pady=(30, 0))

    def build_register(self):
        ttk.Label(self.register_tab, text="Student Registration", style="Title.TLabel").pack(anchor="w", pady=(0, 14))
        form = ttk.Frame(self.register_tab)
        form.pack(anchor="w", fill="x")
        self.reg_id = tk.StringVar()
        self.reg_name = tk.StringVar()
        self.reg_dept = tk.StringVar()
        self.reg_year = tk.StringVar()
        fields = [("Student ID", self.reg_id), ("Full name", self.reg_name),
                  ("Department / course", self.reg_dept), ("Year / semester", self.reg_year)]
        for i, (label, var) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=7, padx=(0, 16))
            ttk.Entry(form, textvariable=var, width=42).grid(row=i, column=1, sticky="w", pady=7)
        ttk.Label(self.register_tab, text=f"Capture {REQUIRED_SAMPLES} clear face samples. Keep your face centered and change your expression/angle slightly.", style="Sub.TLabel").pack(anchor="w", pady=(18, 10))
        ttk.Button(self.register_tab, text="Open camera and capture samples", command=self.register_student).pack(anchor="w")
        self.reg_status = tk.StringVar(value="No student registered yet.")
        ttk.Label(self.register_tab, textvariable=self.reg_status, wraplength=850).pack(anchor="w", pady=14)

    def build_attendance(self):
        ttk.Label(self.attendance_tab, text="Live Face Recognition", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(self.attendance_tab, text="The camera window will open. Press Q in that window to stop attendance.", style="Sub.TLabel").pack(anchor="w", pady=(0, 16))
        buttons = ttk.Frame(self.attendance_tab)
        buttons.pack(anchor="w")
        ttk.Button(buttons, text="Start camera attendance", command=self.start_attendance).pack(side="left", padx=(0, 10))
        ttk.Button(buttons, text="Stop camera", command=self.stop_attendance).pack(side="left")
        self.live_status = tk.StringVar(value="Camera is stopped.")
        ttk.Label(self.attendance_tab, textvariable=self.live_status).pack(anchor="w", pady=18)
        ttk.Label(self.attendance_tab, text="Recognition uses face encodings stored during registration. Unknown faces are not marked present.", style="Sub.TLabel").pack(anchor="w")

    def build_records(self):
        top = ttk.Frame(self.records_tab)
        top.pack(fill="x")
        ttk.Label(top, text="Attendance Records", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_records).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side="right")
        columns = ("id", "name", "date", "time", "status")
        self.tree = ttk.Treeview(self.records_tab, columns=columns, show="headings")
        for col, heading, width in [
            ("id", "Student ID", 130), ("name", "Student Name", 250),
            ("date", "Date", 150), ("time", "Time", 130), ("status", "Status", 120)
        ]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=16)
        ttk.Button(self.records_tab, text="Manage registered students", command=self.manage_students).pack(anchor="w")

    def refresh_all(self):
        students = self.db.students()
        today = date.today().isoformat()
        present = sum(1 for row in self.db.attendance(today, today))
        total = len(self.db.attendance())
        self.student_count.set(str(len(students)))
        self.today_count.set(str(present))
        self.total_count.set(str(total))
        self.known_ids, self.known_encodings = self.db.encodings()
        self.refresh_records()

    def refresh_records(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self.db.attendance():
            self.tree.insert("", "end", values=row)

    def register_student(self):
        sid, name = self.reg_id.get().strip(), self.reg_name.get().strip()
        if not sid or not name:
            messagebox.showwarning("Missing details", "Student ID and full name are required.")
            return
        if any(s[0] == sid for s in self.db.students()):
            messagebox.showerror("Duplicate ID", "This Student ID is already registered.")
            return
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam. Check camera permissions and CAMERA_INDEX.")
            return
        samples = []
        self.reg_status.set("Camera opened. Press SPACE to capture a sample; press Q to cancel.")
        while len(samples) < REQUIRED_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                break
            preview = frame.copy()
            cv2.putText(preview, f"Samples: {len(samples)}/{REQUIRED_SAMPLES} | SPACE capture | Q cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Student Registration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, model="hog")
                encs = face_recognition.face_encodings(rgb, locations)
                if len(encs) != 1:
                    self.reg_status.set("Capture rejected: make sure exactly one face is visible.")
                    continue
                samples.append(encs[0])
                self.reg_status.set(f"Captured {len(samples)} of {REQUIRED_SAMPLES} samples.")
        cap.release()
        cv2.destroyAllWindows()
        if len(samples) != REQUIRED_SAMPLES:
            self.reg_status.set("Registration cancelled. No student record was saved.")
            return
        try:
            self.db.add_student(sid, name, self.reg_dept.get().strip(), self.reg_year.get().strip(), samples)
        except sqlite3.IntegrityError:
            messagebox.showerror("Registration error", "Could not save student. Check whether the ID already exists.")
            return
        self.reg_status.set(f"Successfully registered {name} ({sid}) with {len(samples)} face samples.")
        self.reg_id.set(""); self.reg_name.set(""); self.reg_dept.set(""); self.reg_year.set("")
        self.refresh_all()
        messagebox.showinfo("Success", f"{name} has been registered.")

    def start_attendance(self):
        if not self.known_encodings:
            messagebox.showwarning("No face data", "Register at least one student first.")
            return
        if self.running:
            return
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror("Camera error", "Could not open webcam.")
            return
        self.running = True
        self.live_status.set("Attendance is running. Press Q in the camera window to stop.")
        self.attendance_loop()

    def attendance_loop(self):
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.stop_attendance()
            return
        # Resize for faster processing; scale face coordinates back to the original frame.
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, locations)
        for encoding, (top, right, bottom, left) in zip(encodings, locations):
            label = "Unknown"
            color = (0, 0, 255)
            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best = int(np.argmin(distances))
                if distances[best] <= TOLERANCE:
                    sid = self.known_ids[best]
                    label = sid
                    color = (0, 180, 0)
                    if self.db.mark_present(sid):
                        self.live_status.set(f"Attendance marked for {sid} at {datetime.now().strftime('%H:%M:%S')}.")
                        self.refresh_all()
                    else:
                        self.live_status.set(f"{sid} is already marked present today.")
            top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Live Attendance - press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop_attendance()
            return
        self.root.after(10, self.attendance_loop)

    def stop_attendance(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        self.live_status.set("Camera is stopped.")
        self.refresh_all()

    def export_csv(self):
        rows = self.db.attendance()
        if not rows:
            messagebox.showinfo("No records", "There are no attendance records to export.")
            return
        default = os.path.join(EXPORT_DIR, f"attendance_{date.today().isoformat()}.csv")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=os.path.basename(default),
            initialdir=EXPORT_DIR, filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Student ID", "Student Name", "Date", "Time", "Status"])
            writer.writerows(rows)
        messagebox.showinfo("Export complete", f"Saved attendance report to:\n{path}")

    def manage_students(self):
        win = tk.Toplevel(self.root)
        win.title("Registered Students")
        win.geometry("650x430")
        tree = ttk.Treeview(win, columns=("id", "name", "dept", "year"), show="headings")
        for col, title, width in [("id", "Student ID", 130), ("name", "Name", 210),
                                  ("dept", "Department", 150), ("year", "Year", 100)]:
            tree.heading(col, text=title); tree.column(col, width=width)
        tree.pack(fill="both", expand=True, padx=12, pady=12)
        for row in self.db.students():
            tree.insert("", "end", values=row)

        def delete_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("Select student", "Select a student first.", parent=win)
                return
            row = tree.item(selected[0], "values")
            if messagebox.askyesno("Confirm deletion", f"Delete {row[1]} and their attendance history?", parent=win):
                self.db.delete_student(row[0])
                tree.delete(selected[0])
                self.refresh_all()
        ttk.Button(win, text="Delete selected student", command=delete_selected).pack(pady=(0, 12))

    def on_close(self):
        self.stop_attendance()
        self.db.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
