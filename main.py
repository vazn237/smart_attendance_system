import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import csv
import uuid
import webbrowser

# ==============================
# LOAD FACES
# ==============================

known_encodings = []
known_names = []

if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

for file in os.listdir("known_faces"):
    image_path = os.path.join("known_faces", file)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])


# ==============================
# GUI CLASS
# ==============================

class SmartAttendanceApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System - Elite Prototype")
        self.root.geometry("1200x750")
        self.root.configure(bg="#121212")

        self.video = None
        self.running = False
        self.session_id = None
        self.marked_students = set()

        self.build_layout()

    # ==========================
    # BUILD UI
    # ==========================

    def build_layout(self):

        # LEFT PANEL
        control_frame = tk.Frame(self.root, bg="#1f1f1f", width=300)
        control_frame.pack(side="left", fill="y")

        tk.Label(control_frame, text="SMART ATTENDANCE",
                 font=("Arial", 16, "bold"),
                 fg="cyan", bg="#1f1f1f").pack(pady=15)

        tk.Label(control_frame, text="Subject",
                 fg="white", bg="#1f1f1f").pack()

        self.subject_entry = tk.Entry(control_frame)
        self.subject_entry.pack(pady=5)

        tk.Label(control_frame, text="Student Name (Enroll)",
                 fg="white", bg="#1f1f1f").pack(pady=5)

        self.enroll_entry = tk.Entry(control_frame)
        self.enroll_entry.pack()

        tk.Button(control_frame, text="Start Session",
                  command=self.start_session,
                  bg="green", fg="white", width=18).pack(pady=5)

        tk.Button(control_frame, text="Stop Session",
                  command=self.stop_session,
                  bg="red", fg="white", width=18).pack(pady=5)

        tk.Button(control_frame, text="Enroll Student",
                  command=self.enroll_student,
                  bg="blue", fg="white", width=18).pack(pady=5)

        tk.Button(control_frame, text="Export Attendance",
                  command=self.export_csv,
                  bg="orange", fg="black", width=18).pack(pady=10)

        self.status_label = tk.Label(control_frame,
                                     text="System Status: IDLE",
                                     fg="yellow", bg="#1f1f1f")
        self.status_label.pack(pady=10)

        self.clock_label = tk.Label(control_frame,
                                    fg="white", bg="#1f1f1f")
        self.clock_label.pack()

        self.update_clock()

        # RIGHT PANEL
        right_frame = tk.Frame(self.root, bg="#121212")
        right_frame.pack(side="right", fill="both", expand=True)

        self.video_label = tk.Label(right_frame)
        self.video_label.pack(pady=10)

        # Attendance Table
        self.table = ttk.Treeview(right_frame,
                                  columns=("Name", "Time", "Confidence"),
                                  show="headings")

        self.table.heading("Name", text="Name")
        self.table.heading("Time", text="Time")
        self.table.heading("Confidence", text="Confidence (%)")

        self.table.pack(fill="both", expand=True)

    # ==========================
    # CLOCK
    # ==========================

    def update_clock(self):
        now = datetime.now().strftime("%H:%M:%S")
        self.clock_label.config(text=f"Time: {now}")
        self.root.after(1000, self.update_clock)

    # ==========================
    # START SESSION
    # ==========================

    def start_session(self):
        subject = self.subject_entry.get().strip()

        if not subject:
            messagebox.showerror("Error", "Enter subject name.")
            return

        self.session_id = str(uuid.uuid4())[:8]
        self.marked_students = set()
        self.running = True
        self.subject = subject

        self.status_label.config(text="System Status: RUNNING", fg="lime")
        self.video = cv2.VideoCapture(0)
        self.update_frame()

    # ==========================
    # STOP SESSION
    # ==========================

    def stop_session(self):
        self.running = False
        if self.video:
            self.video.release()

        self.status_label.config(text="System Status: IDLE", fg="yellow")

        messagebox.showinfo("Session Ended",
                            f"Total Marked: {len(self.marked_students)}")

    # ==========================
    # ENROLL
    # ==========================

    def enroll_student(self):
        name = self.enroll_entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Enter student name to enroll.")
            return

        file_path = os.path.join("known_faces", f"{name}.jpg")

        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Enrollment",
                            "Press SPACE to capture.\nESC to cancel.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Enroll Student", frame)
            key = cv2.waitKey(1)

            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == 32:
                cv2.imwrite(file_path, frame)
                break

        cap.release()
        cv2.destroyAllWindows()

        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            messagebox.showinfo("Success", f"{name} enrolled.")
        else:
            messagebox.showerror("Error", "Face not detected.")

    # ==========================
    # UPDATE CAMERA
    # ==========================

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.video.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            confidence = 0

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]

                if matches[best_match_index] and face_distances[best_match_index] < 0.55:
                    name = known_names[best_match_index]

            if name != "Unknown" and name not in self.marked_students:
                now = datetime.now()

                with open("attendance.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.session_id,
                        name,
                        self.subject,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        round(confidence * 100, 2)
                    ])

                self.marked_students.add(name)

                self.table.insert("", "end",
                                  values=(name,
                                          now.strftime("%H:%M:%S"),
                                          round(confidence * 100, 2)))

            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame,
                        f"{name} ({round(confidence*100,2)}%)",
                        (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, box_color, 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # ==========================
    # EXPORT CSV
    # ==========================

    def export_csv(self):
        if os.path.exists("attendance.csv"):
            webbrowser.open("attendance.csv")
        else:
            messagebox.showerror("Error", "No attendance file found.")


# ==============================
# RUN APP
# ==============================

root = tk.Tk()
app = SmartAttendanceApp(root)
root.mainloop()
