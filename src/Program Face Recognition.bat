@echo off
REM ============================
REM 1. Install dependensi via pip
REM ============================
echo Installing required Python packages...
pip install --quiet flask opencv-python numpy matplotlib

REM ============================
REM 2. Jalankan Flask app (app.py) di jendela terpisah
REM ============================
echo Starting Flask server...
start "" python app.py

REM Beri waktu sekitar 2 detik agar server Flask dapat startup dulu
timeout /t 2 /nobreak >nul

REM ============================
REM 3. Buka browser ke http://localhost:5000/
REM ============================
echo Opening browser at http://localhost:5000/ ...
start "" "http://localhost:5000/"

REM Selesai. Jendela .bat ini bisa dibiarkan terbuka atau ditutup secara otomatis:
exit
