# 软件说明
为提升个人工作效率，放置中文视频或音频到指定位置，能一键输出三种字幕 + 定制声线的英文配音。  
仅 Windows 系统使用。

# 使用说明
1. 视频或者音频放入 `start` 文件夹（若没有，在根目录手动创建）。
2. 双击 `agent.bat`，等待运行后在 `output` 文件夹获得中文字幕、英文字幕、双语字幕。
3. 双击 `agent2.bat`，等待运行后在 `output` 文件夹获得英文配音。  
   前提：`output` 文件夹要有英文的 `.srt` 字幕文件，且 `ref-audio` 要有一个参考音色。
4. 双击 `all.bat`，会按序执行第二步和第三步，最终获得三个字幕和一个英语配音。
5. 程序运行开始选择在线/本地，5秒未选择自动选择方式1在线，若在线识别不生效自动切换本地识别。
6. 运行完成之后output文件夹自动生成说明文件，说明最终使用的识别方式。
7. 在线 API 使用两个独立配置文件：`config/online_asr.json`（在线识别）和 `config/online_translate.json`（在线翻译），请分别填写各自的密钥与参数。

# 其他说明
0. 模型放到 `models` 文件夹（若没有，在根目录手动创建），下载 `indexTTS2`、`whisper`、`hf_cache`；依赖环境解压到根目录。
1. `ref-audio` 放置参考音色的音频文件，根目录下仅放置一个即可，如 `ref-audio\TINY.wav'。
2. 双击运行 `cleanup_start.bat`，会清理 `start` 文件夹内所有文件。
3. 双击运行 `cleanup_workspace_full.bat`，会清理 `output` 生成的文件和其他缓存。
4. 如果 `start` 放入的是视频文件，项目根目录会自动生成一个音频文件，不要动它。等程序运行完毕，可按第 3 步进行清理。
5. 程序运行过程中会弹出监控窗口和 CMD 窗口，不要关闭，运行完毕会自动关闭。
6. 批量处理与单个处理方法一致，只需多个文件放到 `start` 文件夹即可。
7. 在线识别失败转为本地识别。

# 模型下载地址
https://pan.baidu.com/s/1CSM6VnBOkQfOyrdT6Ab2Uw?pwd=pyix 提取码: pyix 
默认情况下，程序未读取到模型会自动下载(强烈建议网盘下载)。

# 环境依赖下载地址（必需）
https://pan.baidu.com/s/1mwViPgIuEj4TpzRc5nMYEg?pwd=2wxa 提取码: 2wxa 

