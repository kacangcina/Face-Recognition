# face_recognition.py

import os
import cv2
import numpy as np
import time

def compute_covariance_matrix(X):
    # Menghitung matriks kovarians dari data X (baris-barisnya adalah vektor gambar).
    # Mengembalikan (cov_matrix, mean_vector).
    mean = np.mean(X, axis=0)
    centered = X - mean
    cov = np.dot(centered.T, centered) / (X.shape[0] - 1)
    return cov, mean

def manual_eigen_decomposition(matrix, num_components=10, max_iter=1000, epsilon=1e-10):
    # Melakukan dekomposisi eigen secara manual (power iteration + deflation).
    # Mengembalikan (eigenvalues, eigenvectors) di mana eigenvectors.T adalah shape (n, num_components).
    n = matrix.shape[0]
    eigenvalues = []
    eigenvectors = []
    A = np.array(matrix, copy=True)

    for _ in range(num_components):
        # Inisialisasi vektor b secara acak
        b = np.random.rand(n)
        b = b / np.linalg.norm(b)

        # Power iteration sampai konvergen atau max_iter
        for _ in range(max_iter):
            b_new = np.dot(A, b)
            b_new_norm = np.linalg.norm(b_new)
            if b_new_norm == 0:
                break
            b_new = b_new / b_new_norm
            if np.linalg.norm(b - b_new) < epsilon:
                break
            b = b_new

        # Rayleigh quotient sebagai eigenvalue
        lambda_val = np.dot(b.T, np.dot(A, b))
        eigenvalues.append(lambda_val)
        eigenvectors.append(b)

        # Deflasi: kurangi A dengan komponen eigen ini
        A = A - lambda_val * np.outer(b, b)

    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  # kolom‐kolomnya adalah eigenvector
    return eigenvalues, eigenvectors

def load_images_from_folder(folder, size=(100,100)):
    # Membaca semua file gambar (jpg/png) di dalam folder (tidak rekursif),
    # mengubah ke grayscale, resize, lalu flatten. 
    # Mengembalikan (images_array, labels_list).
    images = []
    labels = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        # Cek apakah itu file gambar (boleh tambahkan ekstensi lain jika diperlukan)
        ext = fname.lower().split('.')[-1]
        if ext not in ["jpg", "jpeg", "png", "bmp"]:
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, size)
        images.append(img_resized.flatten())
        labels.append(fname)
    return np.array(images), labels

def project_face(face_flat, mean_face, eigenfaces):
    # Proyeksikan wajah (1D array) ke ruang Eigenface.
    return np.dot(eigenfaces.T, face_flat - mean_face)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def run_recognition(dataset_folder, input_image_path, output_folder,
                    img_size=(100,100), num_eigen=10):
    
    # 1) Load semua gambar di dataset_folder (folder sudah hasil ekstrak).
    # 2) Hitung mean face dan covariance matrix.
    # 3) Hitung eigenfaces (num_eigen komponen).
    # 4) Proyeksikan semua wajah dataset.
    # 5) Proyeksikan gambar input, hitung jarak Euclidean.
    # 6) Temukan indeks wajah dataset dengan jarak terkecil.
    # 7) Simpan input.jpg dan matched.jpg ke output_folder.
    # 8) Kembalikan matched_label, running_time, relative path input & matched.
    
    start_time = time.time()

    # 1) Muat dataset
    images, labels = load_images_from_folder(dataset_folder, size=img_size)
    if images.size == 0:
        raise ValueError(f"Tidak ada gambar valid di folder dataset: {dataset_folder}")

    # 2) Hitung mean face dan kovarians
    cov_matrix, mean_face = compute_covariance_matrix(images)

    # 3) Hitung eigenvalues & eigenvectors (eigenfaces)
    eigenvalues, eigenvectors = manual_eigen_decomposition(cov_matrix, num_components=num_eigen)
    eigenfaces = eigenvectors  # shape = (pixel_count, num_eigen)

    # 4) Proyeksikan semua gambar dataset
    projections = np.array([project_face(img, mean_face, eigenfaces) for img in images])

    # 5) Baca gambar input
    img_input = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img_input is None:
        raise ValueError(f"Gagal membaca gambar input: {input_image_path}")
    img_input_resized = cv2.resize(img_input, img_size).flatten()
    input_proj = project_face(img_input_resized, mean_face, eigenfaces)

    # 6) Hitung jarak Eucldiean ke setiap proyeksi
    distances = [euclidean_distance(input_proj, p) for p in projections]
    min_idx = int(np.argmin(distances))
    matched_label = labels[min_idx]
    matched_img_flat = images[min_idx]  # vektor piksel grayscale (flatten)

    running_time = time.time() - start_time

    # 7) Simpan input & matched ke output_folder
    #    Output_folder diasumsikan sudah ada (di‐create di app.py)
    #    Simpan input sebagai “input.jpg” dan matched sebagai “matched.jpg”

    # Pastikan output_folder ada
    os.makedirs(output_folder, exist_ok=True)

    # Simpan gambar input (rezise 100x100)
    input_save_name = "input.jpg"
    input_save_path = os.path.join(output_folder, input_save_name)
    # img_input_resized berbentuk flattened (100*100), reshape dulu
    cv2.imwrite(input_save_path, img_input_resized.reshape(img_size))

    # Simpan gambar matched (rezise 100x100)
    matched_save_name = "matched.jpg"
    matched_save_path = os.path.join(output_folder, matched_save_name)
    cv2.imwrite(matched_save_path, matched_img_flat.reshape(img_size))

    # 8) Buat relative path terhadap folder “static”
    #    Misalnya output_folder = "static/uploads/<UUID>"
    #    Maka rel_folder = "uploads/<UUID>"
    rel_folder = os.path.relpath(output_folder, "static").replace("\\", "/")
    saved_input_rel = f"{rel_folder}/{input_save_name}"
    saved_matched_rel = f"{rel_folder}/{matched_save_name}"

    return matched_label, running_time, saved_input_rel, saved_matched_rel
