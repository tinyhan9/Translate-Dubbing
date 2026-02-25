# 软件说明
为提升个人工作效率，放置中文视频或音频到指定位置，能一键输出三种字幕 + 定制声线的英文配音。  
仅 Windows 系统使用。

# 使用说明
1. 视频或者音频放入 `start` 文件夹。
2. 双击 `agent.bat`，等待运行后在 `output` 文件夹获得中文字幕、英文字幕、双语字幕。
3. 双击 `agent2.bat`，等待运行后在 `output` 文件夹获得英文配音。  
   前提：`output` 文件夹要有英文的 `.srt` 字幕文件，且 `ref-audio` 要有一个参考音色。
4. 双击 `all.bat`，会按序执行第二步和第三步，最终获得三个字幕和一个英语配音。

# 其他说明
0. 模型放到 `models` 文件夹，下载 `indexTTS2`、`whisper`、`hf_cache`。
1. `ref-audio` 放置参考音色的音频文件，根目录下仅放置一个即可，如 `ref-audio\\TINY.wav`。
2. 双击运行 `cleanup_start.bat`，会清理 `start` 文件夹内所有文件。
3. 双击运行 `cleanup_workspace_full.bat`，会清理 `output` 生成的文件和其他缓存。
4. 如果 `start` 放入的是视频文件，项目根目录会自动生成一个音频文件，不要动它。等程序运行完毕，可按第 3 步进行清理。
5. 程序运行过程中会弹出监控窗口和 CMD 窗口，不要关闭，运行完毕会自动关闭。
6. 批量处理与单个处理方法一致，只需多个文件放到 `start` 文件夹即可。

# 模型下载地址
https://pan.baidu.com/s/1TgxvKX5LQL4iV_ROVr2UFA?pwd=8mzd  
默认情况下，程序未读取到模型会自动下载。
