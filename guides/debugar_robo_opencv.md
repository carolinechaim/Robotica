 1256  sudo apt install ros-melodic-video-stream-opencv
 1257  roscore
 1258  cd
 1259  nano .bashrc
 1260  source ~/.bashrc
 1261  roscore
 1262  rospack find ros-melodic-video-stream-opencv
 1263  rospack find video-stream-opencv
 1264  rospack 
 1265  rospack list
 1266  rospack list | grep -i video
 1267  rospack find video_stream_opencv
 1268  rosrun video_stream_opencv test_video_resource.py 0
 1269  rosrun video_stream_opencv  test_video_resource.py rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov


rosrun video_stream_opencv test_video_resource.py 0 _width:=640 _height:=480