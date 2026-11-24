# Advanced-Machine-Learning-Project

11.24日小结：

data/arxiv_corpus.json只给出了抓取得到的前200条文献信息，原本是600条的，但是文件太大了，接近2M，就砍了一大半

data/train_tasks.json中给出了根据corpus文件得到的三种难度的20个任务

evaluation/judge.py中调用了`deepseek-v4-pro`对综述结果进行评分
