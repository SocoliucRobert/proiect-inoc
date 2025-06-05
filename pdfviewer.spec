# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\pose_landmark\\pose_landmark_cpu.binarypb', 'mediapipe\\modules\\pose_landmark'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\hand_landmark\\hand_landmark_tracking_cpu.binarypb', 'mediapipe\\modules\\hand_landmark'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\pose_detection\\pose_detection.tflite', 'mediapipe\\modules\\pose_detection'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\pose_landmark\\pose_landmark_full.tflite', 'mediapipe\\modules\\pose_landmark'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\hand_landmark\\hand_landmark_full.tflite', 'mediapipe\\modules\\hand_landmark'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\palm_detection\\palm_detection_full.tflite', 'mediapipe\\modules\\palm_detection'), ('ebook_env\\Lib\\site-packages\\mediapipe\\modules\\hand_landmark\\handedness.txt', 'mediapipe\\modules\\hand_landmark')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pdfviewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
