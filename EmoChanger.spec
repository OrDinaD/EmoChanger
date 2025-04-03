# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data', 'data'),
        ('src/ui/logo.png', '.'),
        ('logs', 'logs'),
    ],
    hiddenimports=['torch', 'torchaudio', 'numpy', 'soundfile', 'PyQt6', 'sys', 'traceback', 'logging'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

app = BUNDLE(
    exe := EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='EmoChanger',
        debug=True,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=True,
        target_arch='arm64',
        codesign_identity=None,
        entitlements_file=None,
        icon='src/ui/logo.png'
    ),
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EmoChanger.app',
    icon='src/ui/logo.png',
    bundle_identifier='com.emochanger.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleDisplayName': 'EmoChanger',
        'NSRequiresAquaSystemAppearance': False,
        'LSBackgroundOnly': False,
        'LSRequiresNativeExecution': True,
    },
)
