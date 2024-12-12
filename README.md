# NDI Viewer

A lightweight and stable NDI viewer application for macOS that allows you to view multiple NDI video sources with minimal resource usage.

## Features

- View multiple NDI sources simultaneously
- Low resource usage (10 FPS default)
- Automatic source discovery
- Stable performance with automatic reconnection
- Clear status indicators and FPS display
- Simple and intuitive interface

## Requirements

- macOS
- Python 3.12 or later
- NDI SDK for macOS (included)

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. The NDI library (`libndi_advanced.dylib`) is included in the repository.

## Usage

1. Run the viewer:
```bash
python3 ndi_viewer.py
```

2. Using the viewer:
   - Click "Refresh" to scan for NDI sources
   - Select a source from the dropdown menu
   - Click "Connect" to view the stream
   - The status bar shows connection state and FPS
   - Click "Disconnect" to stop viewing

## Performance Settings

Default settings are optimized for stability:
- Frame rate: 10 FPS
- Resolution: 640x480
- CPU processing only

## Troubleshooting

If you experience issues:
1. Check the `ndi_viewer.log` file for error messages
2. Ensure NDI sources are running on your network
3. Try disconnecting and reconnecting to the source
4. Restart the application if needed

## Files

- `ndi_viewer.py` - Main application
- `libndi_advanced.dylib` - NDI library for macOS
- `requirements.txt` - Python dependencies
- `ndi_viewer.log` - Application log file

## License

This project uses the NDI SDK, which is subject to the NDI SDK License Agreement.
