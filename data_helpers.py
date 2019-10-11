# coding = utf-8
import string

def preprocess(data_path, src_path, tgt_path):
    """
    数据预处理。将 PART_* 系列文件处理后输出为短文本(.src)，摘要(.tgt)
    Args:
        data_path str 源数据路径
        src_path str 短文本路径
        tgt_path str 摘要路径
    """
    src_list = []
    tgt_list = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        is_sum = 0
        is_short = 0
        is_avaliable = 1
        for line in lines:
            line = line.strip()
            if line.find("<human_label>") != -1:
                score = line[13:14]
                if int(score) < 3:
                    is_avaliable = 0
                else:
                    is_avaliable = 1
            if is_avaliable == 1:
                if is_sum == 1:
                    tgt_list.append(line+"\n")
                    is_sum = 0
                    is_short = 0
                if is_short == 1:
                    src_list.append(line+"\n")
                    is_sum = 0
                    is_short = 0
                if line.find("<summary>") != -1:
                    is_sum = 1
                if line.find("<short_text>") != -1:
                    is_short = 1


    with open(src_path, 'w', encoding='utf-8') as f:
        f.writelines(src_list)

    with open(tgt_path, 'w', encoding='utf-8') as f:
        f.writelines(tgt_list)


if __name__ == "__main__":
    train_data_path = './DATA/PART_I.txt'
    train_src_path = './res/train.src'
    train_tgt_path = './res/train.tgt'
    valid_data_path = './DATA/PART_II.txt'
    valid_src_path = './res/valid.src'
    valid_tgt_path = './res/valid.tgt'
    test_data_path = './DATA/PART_III.txt'
    test_src_path = './res/test.src'
    test_tgt_path = './res/test.tgt'
    preprocess(train_data_path, train_src_path, train_tgt_path)
    preprocess(valid_data_path, valid_src_path, valid_tgt_path)
    preprocess(test_data_path, test_src_path, test_tgt_path)
