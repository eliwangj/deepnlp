import sys
sys.path.append("../..")
from DeepNLP.model.DSeg import DSeg

if __name__ == '__name__':
    # joint chinese word segmentation
    model_path='/data/Yangyang/D_project/saved_models/Seg_bilstm_cn_2021-07-11-18-07-29/model'
    deepnlp = DSeg.load_model(model_path=model_path, no_cuda=False)
    sentence = [
        ['法', '正', '研', '究', '从', '波', '黑', '撤', '军', '计', '划', '新', '华', '社', '巴', '黎', '９', '月', '１', '日', '电', '（',
         '记', '者', '张', '有', '浩', '）'],
        ['法', '国', '国', '防', '部', '长', '莱', '奥', '塔', '尔', '１', '日', '说', '，法', '国', '正', '在', '研', '究', '从', '波', '黑',
         '撤', '军', '的', '计', '划', '。']]
    result_list = deepnlp.predict(sentence_list=sentence)
    # print(result_list)
    #[['法', '正', '研究', '从', '波黑', '撤军', '计划', '新华社', '巴黎', '９月', '１日', '电', '（', '记者', '张有浩', '）'],
    # ['法国', '国防', '部长', '莱奥塔尔', '１日', '说', '，法国', '正在', '研究', '从', '波黑', '撤军', '的', '计划', '。']]
