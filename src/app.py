import os
import zipfile
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash
from face_recognition import run_recognition

app = Flask(__name__)
app.secret_key = "ganti_Ini_Dengan_Secret_Key_Panas_dan_Random"  # minimal 16 karakter

# 1. Tentukan BASE_UPLOAD_FOLDER di mana Flask akan menyimpan file upload & hasil ekstrak
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    #Render form ‘index.html’ yang masih kosong (belum ada hasil).
    
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    # Tangani form POST dari index.html.
    # 1) Validasi keberadaan file dataset_zip & image_input.
    # 2) Simpan file ZIP & gambar input ke folder session unik.
    # 3) Ekstrak ZIP dataset ke folder ‘dataset/’ di dalam session folder.
    # 4) Panggil run_recognition() untuk proses eigenface.
    # 5) Terima hasil, buat URL statis untuk gambar, lalu render kembali index.html.
    
    if "dataset_zip" not in request.files or "image_input" not in request.files:
        flash("Anda harus mengunggah file dataset (ZIP) dan file gambar input.", "error")
        return redirect(url_for("index"))

    dataset_zip = request.files["dataset_zip"]
    image_input = request.files["image_input"]

    # Validasi: keduanya wajib diisi
    if dataset_zip.filename == "" or image_input.filename == "":
        flash("Silakan pilih file dataset (ZIP) dan file gambar input.", "error")
        return redirect(url_for("index"))

    # 2) Buat folder session unik (UUID)
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)

    # 3) Simpan ZIP dataset
    zip_path = os.path.join(session_folder, "dataset.zip")
    dataset_zip.save(zip_path)

    # 4) Ekstrak ZIP ke folder session_folder/dataset
    extract_folder = os.path.join(session_folder, "dataset")
    os.makedirs(extract_folder, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as uzip:
            uzip.extractall(extract_folder)
    except zipfile.BadZipFile:
        flash("File ZIP tidak valid atau rusak.", "error")
        return redirect(url_for("index"))

    # 5) Simpan file gambar input
    ext = os.path.splitext(image_input.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        # paksa menjadi .jpg
        ext = ".jpg"
    input_filename = "input" + ext
    input_save_path = os.path.join(session_folder, input_filename)
    image_input.save(input_save_path)

    # 6) Panggil run_recognition
    try:
        matched_label, running_time, input_rel, matched_rel = run_recognition(
            dataset_folder   = extract_folder,
            input_image_path = input_save_path,
            output_folder    = session_folder,
            img_size         = (100, 100),
            num_eigen        = 10
        )
    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses: {e}", "error")
        return redirect(url_for("index"))

    # 7) Buat URL statis untuk gambar (relatif ke folder static/)
    input_image_url   = url_for("static", filename=input_rel)
    matched_image_url = url_for("static", filename=matched_rel)

    # 8) Render ulang index.html dengan hasil
    return render_template(
        "index.html",
        matched_label      = matched_label,
        running_time       = running_time,
        input_image_url    = input_image_url,
        matched_image_url  = matched_image_url
    )


if __name__ == "__main__":
    # Jalankan di http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
