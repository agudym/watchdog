<p align="center">
  <h1 align="center">The Watchdog Never Sleeps! </h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/anton-gudym-829772233/">Anton Gudym</a>
  </p>
</p>
<p align="center">
    <img src="assets/watchdog.jpeg" alt="example" width=70%>
    <br>
    Light-weight framework for Objects AI-detection with Live-Cameras (USB/IP) and Telegram-bot notifications. Use Yolo or adjust for you own AI-models support and catch the best shot!
</p>

## Installation

Installation for Linux, Win or Mac is similar. Jetson TX1/2 (Ubuntu18/Arm/Python3.6) works too! The three main steps are below:

1. Download the watchdog and install required python-modules (linux-shell command):
```shell
cd /home/user
git clone https://github.com/agudym/watchdog.git
cd watchdog
python -m pip install -r requirements.txt
```

2. Get AI-detector and weights/configuration, YOLOv6 is supported at the moment. Or simply adjust `detector.py` for your model.<br>
[Optional] For GPU(Nvidia CUDA) accelerated detection with Torch follow <a href="https://pytorch.org/get-started/locally/">instructions</a>. <br>
[Optional] For Jetson TX1/2 with hardware accelerated camera's stream decoding build OpenCV with GStreamer using <a href="https://gist.github.com/mtc-20/c1f324f70fad774ca6f381c07cb3f19a">instructions</a>.
```shell
cd ..
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt
```
Add 2 new paths to `config.json` (see below an example). Then run `python watchdog/detector.py config.json <path to test image dir>` to verify that detection works.

3. [Optional] Register your watchdog-bot in Telegram: basically, send <a href="https://telegram.me/BotFather">@BotFather</a> the command `/newbot` to get token-string (keep it SECRET!).
<br>
3.1. Start conversation with your new bot in Telegram (it won't respond so far, though).
<br>
3.2. Copy-paste the token to the configuration file `config.json`.
<br>
3.3. Run `python watchdog/bot.py configs/config.json` to initialize chat-id in the config.

Alright, the environment is set!

## Cameras configration
Last but not least is to **configure the cameras**. USB or IP, one or many - doesn't matter. First of all it's recommended to use <a href="https://www.videolan.org/vlc/">VLC</a> to check the camera (ip-connection). Then fill the missing fields in `config.json`.
<br>
Below is the complete example. Follow the descriptions there to setup your own configuration and run `python watchdog/camera.py configs/config.json` to make sure that images are captured correctly.

## Start
Execute something like:
```shell
cd /home/user/watchdog
python start_watchdog.py configs/config.json
```
Or run a detached process, e.g. via temporary shell ssh-connection (linux shell command):
```shell
nohup python start_watchdog.py configs/config.json &> log_nohup.txt &
```
That's it! Now watchdog will record activity and Telegram-bot will speak and report!

## Configration example
```json
// Comments must be removed before use!
{
    "version": "0.1.0",
    "Watchdog":
    {
        // Output LOCAL directory with recorded images and logs
        "output_path" : "/home/user/recordings",
        // Each camera save jpg images with given frequency (even if nothing is detected), in seconds
        "img_log_timeout" : 300,
        // Log everything to the file
        "verbose" : true
    },
    "Detector" :
    {
        "Init" :
        {
            // Path to the Yolo library (current version doesn't support installation)
            "yolo_lib_path" : "/home/user/YOLOv6",
            // Path to the model weights
            "checkpoint_path" : "/home/user/YOLOv6/yolov6n.pt",
            // Resolution used for faster inference, so far it's (1920, 1080) * k, k = 1024 / 1920.
            // Adjust if cameras have another aspect (padding might be needed, e.g. w/h mod 32 == 0)
            "img_width_height_channels" : [1024, 576, 3],
            // Processing unit: auto, cpu, cuda
            "device_switch" : "auto",
            // Switch to lighter (fp32 -> fp16) computations (if possible)
            "is_model_fp16" : false
        },
        // How confident the detector must be to signal/alarm with the bot, 0 < confidence_threshold < 1
        "confidence_threshold" : 0.8,
        // Which types of objects to detect, see detector.py for the list
        "categories" : ["person", "bird", "cat", "dog"],
        // Merge close bounding boxes represented with normalized 4-vector (x_low, y_low, x_high, y_high) in [0,1]^4
        "bbox_merge_dist" : 0.01
    },
    "Bot" :
    {
        // How often the watchdog can bother via Telegram
        "bot_warning_timeout" : 30,
        // The SECRET! token received from the @BotFather
        "token" : "1234567890:AaBbCcDdEeFfGg123AaBbCcDdEeFfGg",
        // Your bot-chat identifier (aquired from bot.py)
        "chat_id" : "987654321"
    },
    "Cameras":
    [
        // Main camera setting, it's address in the system/network (supported by OpenCV's cv::VideoCapture):
        // 1. USB camera integer-index, e.g. integer 0 or "/dev/video0"
        // 2. RTSP address, e.g. "rtsp://192.168.1.101"
        // 3. GStreamer pipeline
        // 4. OR! PATH to an existing DIRECTORY with Images (Good for debugging)
        {
            // USB camera
            "uri": 0,
            // Camera alias for logging, e.g. "cam1", "cam_home", "cam_street"
            "name": "usb0",
            // Number of frames per second (for data-polling and cpu-resource preservation)
            "fps": 30.0,
            // Initialize, capture, release time limit in seconds (before reset)
            "timeout_err": 30,
            // Number of failed frames before reset
            "count_err" : 100
        },
        {
            // The most universal approach for IP cameras supporting RTSP
            "uri": "rtsp://192.168.1.101",
            "name": "ip1",
            "fps": 15.0,
            "timeout_err": 30,
            "count_err" : 100
        },
        {
            // Manually set GStreamer-pipeline for "older" camera with h264 stream for Jetson TX1/2 (hardware decoding without buffering)
            "uri": "rtspsrc location=rtsp://192.168.1.101 ! rtph264depay ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=True",
            "name": "ip2_h264",
            "fps": 25.0,
            "timeout_err": 30,
            "count_err" : 100
        },
        {
            // Manually set GStreamer-pipeline for h265 stream for Jetson TX1/2
            "uri": "rtspsrc location=rtsp://192.168.1.102 ! rtph265depay ! decodebin ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=True",
            "name": "ip2_h265",
            "fps": 25.0,
            "timeout_err": 30,
            "count_err" : 100
        }
        // more cams go here ...
    ]
}
```
Setting up hardware decoding could be tricky (especially for Jetson) but necessary to **avoid lags or delays** during streams processing. Use `gstreamer` to find the right pipeline, e.g.:
```shell
gst-launch-1.0 rtspsrc location=rtsp://192.168.1.101 ! rtph264depay ! nvv4l2decoder ! nvvidconv ! video/x-raw, 'width=(int)1920, height=(int)1080, format=(string)BGRx' ! videoconvert ! nv3dsink max-buffers=1 drop=True
```
## License

The library and it's sources are released under the [MIT License](./LICENSE).
