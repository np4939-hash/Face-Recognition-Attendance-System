import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import csv
import datetime
import cv2
import numpy as np

# -------------------- Folders and files --------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "AttendanceData"
STUDENT_DIR = DATA_DIR / "StudentDetails"
IMAGE_DIR = DATA_DIR / "TrainingImage"
MODEL_DIR = DATA_DIR / "Training"
ATTENDANCE_DIR = DATA_DIR / "Attendance"

STUDENT_FILE = STUDENT_DIR / "StudentDetails.csv"
MODEL_FILE = MODEL_DIR / "trainer.yml"

for folder in (STUDENT_DIR, IMAGE_DIR, MODEL_DIR, ATTENDANCE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Use OpenCV's built-in Haar Cascade file.
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(str(CASCADE_FILE))

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV's face detector.")

# -------------------- App state --------------------

camera = None
camera_job = None
capture_count = 0
capture_id = ""
capture_name = ""
recognizer = None
seen_today = set()

# -------------------- Student data helpers --------------------

def load_students():
    students = {}
    if STUDENT_FILE.exists():
        with STUDENT_FILE.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                student_id = row.get("ID", "").strip()
                name = row.get("NAME", "").strip()
                if student_id and name:
                    students[student_id] = name
    return students


def save_student(student_id, name):
    students = load_students()
    if student_id not in students:
        write_header = not STUDENT_FILE.exists()
        with STUDENT_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["ID", "NAME"])
            writer.writerow([student_id, name])


def refresh_student_count():
    count = len(load_students())
    count_label.config(text=f"Total registrations: {count}")


def show_message(text):
    status_label.config(text=text)


# -------------------- Camera helpers --------------------

def open_camera():
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        camera = None
        messagebox.showerror(
            "Camera error",
            "Could not open the webcam.\nClose other apps using the camera and try again."
        )
        return False
    return True


def close_camera():
    global camera, camera_job
    if camera_job is not None:
        window.after_cancel(camera_job)
        camera_job = None
    if camera is not None:
        camera.release()
        camera = None
    cv2.destroyAllWindows()


def stop_camera():
    close_camera()
    show_message("Camera stopped.")


# -------------------- Register student --------------------

def take_images():
    global capture_count, capture_id, capture_name

    if camera is not None:
        messagebox.showinfo("Camera busy", "Finish the current camera task first.")
        return

    student_id = id_entry.get().strip()
    name = name_entry.get().strip()

    if not student_id or not name:
        messagebox.showwarning("Missing details", "Enter both student ID and name.")
        return
    if not student_id.isdigit():
        messagebox.showwarning("Invalid ID", "Student ID must contain digits only.")
        return
    if not all(part.isalpha() for part in name.split()):
        messagebox.showwarning("Invalid name", "Enter a name using letters and spaces only.")
        return

    students = load_students()
    if student_id in students:
        messagebox.showwarning("Already registered", "This student ID is already registered.")
        return

    if not open_camera():
        return

    capture_id = student_id
    capture_name = name
    capture_count = 0
    show_message("Look at the camera. Press Q in the camera window to stop early.")
    take_images_frame()


def take_images_frame():
    global camera_job, capture_count

    if camera is None:
        return

    ok, frame = camera.read()
    if not ok:
        stop_camera()
        messagebox.showerror("Camera error", "Could not read a frame from the webcam.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        capture_count += 1

        image_path = IMAGE_DIR / f"{capture_id}.{capture_count}.jpg"
        cv2.imwrite(str(image_path), face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

    cv2.putText(
        frame, f"Images: {capture_count}/40 - press Q to stop",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    cv2.imshow("Register face - press Q to stop", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or capture_count >= 40:
        close_camera()
        if capture_count >= 10:
            save_student(capture_id, capture_name)
            refresh_student_count()
            show_message(f"Captured {capture_count} images for {capture_name}. Now click Save Profile.")
        else:
            # Remove incomplete images so they cannot affect training.
            for path in IMAGE_DIR.glob(f"{capture_id}.*.jpg"):
                path.unlink(missing_ok=True)
            show_message("Not enough face images. Please try again and capture at least 10.")
        return

    camera_job = window.after(10, take_images_frame)


# -------------------- Train face recognition model --------------------

def train_images():
    global recognizer

    if camera is not None:
        messagebox.showinfo("Camera busy", "Stop the camera task before saving the profile.")
        return

    if not hasattr(cv2, "face"):
        messagebox.showerror(
            "OpenCV package missing",
            "Face recognition needs opencv-contrib-python.\n"
            "In your terminal run:\n"
            "python -m pip uninstall opencv-python -y\n"
            "python -m pip install opencv-contrib-python"
        )
        return

    faces = []
    labels = []

    for image_path in IMAGE_DIR.glob("*.jpg"):
        try:
            # File format: studentID.sampleNumber.jpg
            student_id = int(image_path.name.split(".")[0])
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(image)
                labels.append(student_id)
        except (ValueError, IndexError):
            continue

    if not faces:
        messagebox.showwarning("No images", "Register at least one student before saving the profile.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.save(str(MODEL_FILE))

    show_message(f"Profile saved. Students trained: {len(set(labels))}")
    messagebox.showinfo("Success", "Face recognition profile saved successfully.")


# -------------------- Attendance --------------------

def attendance_file():
    today = datetime.date.today().strftime("%Y-%m-%d")
    return ATTENDANCE_DIR / f"Attendance_{today}.csv"


def mark_attendance(student_id, name):
    today_file = attendance_file()
    today = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%H:%M:%S")

    # Avoid recording the same student more than once in one camera session/day.
    if student_id in seen_today:
        return

    file_exists = today_file.exists()
    with today_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["ID", "NAME", "DATE", "TIME"])
        writer.writerow([student_id, name, today, now])

    seen_today.add(student_id)
    attendance_table.insert("", "end", values=(student_id, name, today, now))
    show_message(f"Attendance marked for {name}.")


def load_today_attendance():
    today_file = attendance_file()
    if not today_file.exists():
        return

    with today_file.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            student_id = row.get("ID", "")
            name = row.get("NAME", "")
            date = row.get("DATE", "")
            time_text = row.get("TIME", "")
            if student_id:
                seen_today.add(student_id)
                attendance_table.insert(
                    "", "end", values=(student_id, name, date, time_text)
                )


def track_images():
    global recognizer

    if camera is not None:
        messagebox.showinfo("Camera busy", "Finish the current camera task first.")
        return
    if not MODEL_FILE.exists():
        messagebox.showwarning("Profile missing", "Register students, then click Save Profile first.")
        return
    if not hasattr(cv2, "face"):
        messagebox.showerror("OpenCV package missing", "Install opencv-contrib-python.")
        return

    students = load_students()
    if not students:
        messagebox.showwarning("No students", "Register a student first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    if not open_camera():
        return

    show_message("Attendance camera running. Press Q in the camera window to stop.")
    track_images_frame(students)


def track_images_frame(students):
    global camera_job

    if camera is None:
        return

    ok, frame = camera.read()
    if not ok:
        stop_camera()
        messagebox.showerror("Camera error", "Could not read a frame from the webcam.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        label, confidence = recognizer.predict(face)

        # Lower LBPH confidence values mean a closer match.
        if confidence < 65 and str(label) in students:
            student_id = str(label)
            name = students[student_id]
            mark_attendance(student_id, name)
            display_text = name
            box_color = (0, 180, 0)
        else:
            display_text = "Unknown"
            box_color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            frame, display_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
        )

    cv2.imshow("Taking Attendance - press Q to stop", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        close_camera()
        show_message("Attendance camera stopped.")
        return

    camera_job = window.after(10, lambda: track_images_frame(students))


# -------------------- Buttons --------------------

def clear_fields():
    id_entry.delete(0, tk.END)
    name_entry.delete(0, tk.END)
    show_message("Enter student details to register.")


def on_closing():
    close_camera()
    window.destroy()


# -------------------- GUI --------------------

window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.geometry("1100x680")
window.minsize(900, 600)
window.configure(bg="#244b50")

title = tk.Label(
    window,
    text="Face Recognition Based Attendance System",
    font=("Arial", 22, "bold"),
    bg="#244b50",
    fg="white"
)
title.pack(pady=18)

main_frame = tk.Frame(window, bg="#244b50")
main_frame.pack(fill="both", expand=True, padx=25, pady=10)

register_frame = tk.LabelFrame(
    main_frame, text="Register New Student",
    font=("Arial", 14, "bold"), bg="white", padx=20, pady=20
)
register_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))

attendance_frame = tk.LabelFrame(
    main_frame, text="Mark Student Attendance",
    font=("Arial", 14, "bold"), bg="white", padx=15, pady=20
)
attendance_frame.pack(side="right", fill="both", expand=True, padx=(12, 0))

tk.Label(register_frame, text="Student ID", font=("Arial", 12), bg="white").pack(anchor="w")
id_entry = tk.Entry(register_frame, font=("Arial", 13))
id_entry.pack(fill="x", pady=(5, 15))

tk.Label(register_frame, text="Student Name", font=("Arial", 12), bg="white").pack(anchor="w")
name_entry = tk.Entry(register_frame, font=("Arial", 13))
name_entry.pack(fill="x", pady=(5, 15))

tk.Button(
    register_frame, text="Clear", command=clear_fields,
    font=("Arial", 12), bg="#263c9c", fg="white"
).pack(fill="x", pady=6)

tk.Button(
    register_frame, text="Take Images", command=take_images,
    font=("Arial", 13, "bold"), bg="#00aeef"
).pack(fill="x", pady=6)

tk.Button(
    register_frame, text="Save Profile", command=train_images,
    font=("Arial", 13, "bold"), bg="#00aeef"
).pack(fill="x", pady=6)

count_label = tk.Label(register_frame, text="Total registrations: 0", bg="white", font=("Arial", 12, "bold"))
count_label.pack(pady=20)

status_label = tk.Label(
    register_frame, text="Enter student details to register.",
    bg="white", fg="#174d35", wraplength=350, font=("Arial", 10)
)
status_label.pack(pady=10)

tk.Button(
    attendance_frame, text="Take Attendance", command=track_images,
    font=("Arial", 13, "bold"), bg="#00aeef"
).pack(fill="x", pady=(0, 12))

columns = ("id", "name", "date", "time")
attendance_table = ttk.Treeview(attendance_frame, columns=columns, show="headings", height=16)
for column, heading, width in (
    ("id", "ID", 75),
    ("name", "NAME", 130),
    ("date", "DATE", 100),
    ("time", "TIME", 90),
):
    attendance_table.heading(column, text=heading)
    attendance_table.column(column, width=width, anchor="center")
attendance_table.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(attendance_frame, orient="vertical", command=attendance_table.yview)
attendance_table.configure(yscrollcommand=scrollbar.set)
scrollbar.place(relx=0.98, rely=0.18, relheight=0.76, anchor="ne")

tk.Button(
    window, text="Quit", command=on_closing,
    font=("Arial", 12, "bold"), bg="#263c9c", fg="white", width=15
).pack(pady=12)

refresh_student_count()
load_today_attendance()
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
