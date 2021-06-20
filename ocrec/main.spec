# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['C:/kyun/conda/ocr/main.py'],
             pathex=['C:\\kyun\\conda\\ocr'],
             binaries=[],
             datas=[('C:/kyun/conda/ocr/networks', 'networks/'), ('C:/kyun/conda/ocr/static', 'static/'), ('C:/kyun/conda/ocr', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='C:\\kyun\\conda\\ocr\\icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
